"""
response_generator_v2.py — Unified Response Generation
==========================================================
Merges response_generator.py + baseline_t5_gemma.py + experiment_rag_prompt.py
into a single file with all modes selectable via --mode.

MODES:
  # Baselines
  t5_base        — T5-Base zero-shot (google-t5/t5-base, seq2seq)   [System 1]
  gemma_pt       — Gemma-3-1B PT zero-shot (completion-style)       [System 2]
  all_baselines  — t5_base + gemma_pt

  # RAG / Aspect systems (all use gemma-2-2b-it, completion-style prompts)
  baseline       — Gemma-PT zero-shot, no RAG, no aspects           [alias: gemma_pt]
  rag            — Gemma-PT + RAG only (pure dense retrieval)        [System 3]
  aspect         — Gemma-PT + Aspects only (no RAG)                 [System 4]
  rag_aspect     — Gemma-PT + RAG + Aspects (full system)           [System 5]
  all            — rag + aspect + rag_aspect (no duplicate baseline)

  # Guided prompt experiment (NEW — from experiment_rag_prompt.py)
  rag_guided     — RAG only + explicit use-context instruction
  rag_asp_guided — RAG + Aspects + explicit use-context instruction
  all_guided     — rag_guided + rag_asp_guided

  # Everything
  all_modes      — runs all 9 systems

INDEX RANGE:
  --start_idx / --end_idx : inclusive range filter on idx column
  Supports small test runs e.g. --start_idx 2 --end_idx 8

USAGE:
  python -m src.generation.response_generator_v2 \
    --config config/generation_config.yaml \
    --mode all \
    --input data/processed/cs-support-multi-test.csv \
    --aspects_file output/aspect/aspect_test_v2.csv \
    --start_idx 0 --end_idx 199

  # Quick smoke test on 7 records
  python -m src.generation.response_generator_v2 \
    --config config/generation_config.yaml \
    --mode baseline \
    --input data/processed/cs-support-multi-test.csv \
    --start_idx 2 --end_idx 8
"""

import os
import re
import sys
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

ASPECT_COLS = [
    "prob_sub", "prob_statement", "cause", "intent", "ticket_type_nli",
    # New fields from upgraded aspect_extractor
    "entities", "action_taken", "urgency",
]

# ── ADDITIONAL BASELINE MODEL REGISTRY ───────────────────────────────────────
# Models run with their own load/generate logic (different from main Gemma config).
# Main model: gemma-2-2b-it, loaded from generation_config.yaml.
# Extra models below have their own load/generate logic (T5 seq2seq).
EXTRA_MODEL_REGISTRY = {
    "t5_base": {
        "model_id"      : "google-t5/t5-base",
        "seq2seq"       : True,
        "use_4bit"      : False,
        "torch_dtype"   : "float32",
        "label"         : "T5-Base Zero-Shot",
        "max_new_tokens": 256,
        "is_pretrained" : False,
    },
    "gemma_pt": {
        # Standalone gemma_pt baseline runner — same model as main config.
        # Kept here so --mode gemma_pt runs the dedicated PT zero-shot system.
        "model_id"      : "google/gemma-2-2b-it",
        "seq2seq"       : False,
        "use_4bit"      : True,
        "torch_dtype"   : "bfloat16",
        "label"         : "Gemma-3-1B Pre-Trained (zero-shot)",
        "max_new_tokens": 512,
        "is_pretrained" : True,   # no chat template, PT completion-style prompt
    },
}


def _is_pretrained(model_id: str) -> bool:
    """True for pre-trained (non-instruct) models — affects prompt format."""
    return model_id.endswith("-pt") or "-pt-" in model_id

# Sentiment: lightweight lexicon-based scorer (already in requirements.txt)
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

from src.utils import (
    get_hf_token,
    load_csv_or_excel,
    Checkpointer,
    MetricsCalculator,
    compare_models,
    log,
)
try:
    from system_profiler import Profiler
except ImportError:
    class Profiler:
        def __enter__(self):
            self.result = self._Result()
            return self

        def __exit__(self, *args):
            return None

        class _Result:
            def summary(self):
                return "(system_profiler not installed)"

import yaml
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    # model
    model_id        : str  = "google/gemma-2-2b-it"
    device          : str  = "cuda"
    torch_dtype     : str  = "bfloat16"
    use_4bit        : bool = True
    max_new_tokens  : int  = 200
    do_sample       : bool = False
    # retrieval
    retrieval_config: str  = "config/inference_retrieval_config.yaml"
    top_k           : int  = 3
    # data — primary: raw tickets (subject, body, language); optional gold `answer`
    input_tickets_csv: str = ""
    # optional: pre-extracted aspects CSV (used if input_tickets_csv empty / legacy)
    aspects_csv     : str  = "data/processed/aspects_test.csv"
    # optional: aspects file to merge onto input_tickets_csv (same rows) by idx
    # useful when you keep tickets and extracted aspects in separate files
    aspects_file    : str  = ""
    # run SFT aspect model when aspect columns are missing (needs CUDA)
    extract_aspects : bool = True
    force_reextract_aspects: bool = False
    aspect_extractor_config: str  = ""
    aspect_student_base    : str  = ""
    aspect_adapter         : str  = ""
    aspect_adapter_subfolder: str = ""
    aspect_batch_size      : int  = 0
    aspect_max_seq_len     : int  = 0
    aspect_max_new_tokens  : int  = 0
    body_max_chars  : int  = 0
    # batched generation — RTX 4090 + 1B 4-bit handles 8 comfortably
    generation_batch_size: int = 8
    # output
    output_dir      : str  = "output/generation"
    checkpoint_every: int  = 50
    # metrics
    bertscore_model : str  = "bert-base-multilingual-cased"
    bertscore_device: str  = "cuda"
    bertscore_batch : int  = 32


def load_generation_config(path: str) -> GenerationConfig:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return GenerationConfig(**{k: v for k, v in cfg.items()
                                if k in GenerationConfig.__dataclass_fields__})


def _aspect_columns_ready(df: pd.DataFrame) -> bool:
    return all(c in df.columns for c in ASPECT_COLS)


def _resolve_aspect_extractor_kwargs(config: GenerationConfig) -> Dict:
    from src.aspect import aspect_extractor as ae

    ae_cfg: Dict = {}
    ac = (config.aspect_extractor_config or "").strip()
    if ac and Path(ac).exists():
        ae_cfg = ae.load_config(ac)

    def pick(key_cfg: str, key_ae: str, default):
        v = getattr(config, key_cfg, "") or ""
        if isinstance(v, str) and v.strip():
            return v.strip()
        return ae_cfg.get(key_ae, default)

    student = pick("aspect_student_base", "student_base", ae.STUDENT_BASE_DEFAULT)
    adapter = pick("aspect_adapter",      "adapter",      ae.ADAPTER_DEFAULT)
    subf    = pick("aspect_adapter_subfolder", "adapter_subfolder", ae.ADAPTER_SUBF_DEFAULT)
    batch   = config.aspect_batch_size or int(ae_cfg.get("batch", 0) or 0)
    msl     = config.aspect_max_seq_len or int(ae_cfg.get("max_seq_len", ae.MAX_SEQ_LEN_DEFAULT))
    mnt     = config.aspect_max_new_tokens or int(ae_cfg.get("max_new_tokens", ae.MAX_NEW_TOKENS_DEFAULT))
    return {
        "student_base"     : student,
        "adapter"          : adapter,
        "adapter_subfolder": (subf or None),
        "batch_size"       : batch,
        "max_seq_len"      : msl,
        "max_new_tokens"   : mnt,
    }


def enrich_dataframe_with_aspects(df: pd.DataFrame,
                                   config: GenerationConfig) -> pd.DataFrame:
    """
    Run aspect_extractor SFT model to populate all 5 aspect columns:
      prob_sub, prob_statement, cause, intent, ticket_type_nli

    Skips if extract_aspects=False or columns already present and
    force_reextract_aspects=False.
    """
    if not config.extract_aspects:
        return df
    if not config.force_reextract_aspects and _aspect_columns_ready(df):
        log.info("[Aspects] Using existing aspect columns "
                 "(set force_reextract_aspects=true to refresh)")
        return df

    from src.aspect import aspect_extractor as ae

    log.info("[Aspects] Running SFT aspect extractor (in-process)...")
    miss = [c for c in ("subject", "body") if c not in df.columns]
    if miss:
        raise ValueError(
            f"Need columns {miss} for aspect extraction; got: {list(df.columns)}")

    df = df.copy()
    if "language" not in df.columns:
        df["language"] = "en"
    for c in ASPECT_COLS:
        if c not in df.columns:
            df[c] = ""

    kw = _resolve_aspect_extractor_kwargs(config)
    try:
        profile = ae.get_gpu_profile()
    except RuntimeError as e:
        raise RuntimeError(
            "Aspect extraction requires CUDA. "
            "Use a pre-extracted CSV or set extract_aspects: false."
        ) from e

    batch_size = kw["batch_size"] or profile["batch"]
    tok, model = ae.get_model(
        student_base      = kw["student_base"],
        adapter           = kw["adapter"],
        adapter_subfolder = kw["adapter_subfolder"],
        dtype             = profile["dtype"],
    )

    df    = df.reset_index(drop=True)
    total = len(df)
    t0    = time.time()
    for start in range(0, total, batch_size):
        end        = min(start + batch_size, total)
        sub        = df.iloc[start:end]
        batch_rows = (sub[["subject", "body", "language"]]
                      .fillna("").to_dict(orient="records"))
        preds = ae.extract_batch(
            tok, model, batch_rows,
            max_seq_len    = kw["max_seq_len"],
            max_new_tokens = kw["max_new_tokens"],
            temperature    = 0.0,
        )
        for j, pred in enumerate(preds):
            idx = start + j
            for k in ASPECT_COLS:
                df.at[idx, k] = pred.get(k, "")
        if (end % max(batch_size * 5, 50) == 0) or end == total:
            log.info(f"[Aspects] {end}/{total}  "
                     f"{(time.time() - t0) / 60:.1f} min elapsed")

    log.info("[Aspects] Extraction complete")
    ae._model     = None
    ae._tokenizer = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def resolve_data_path(args, cfg: GenerationConfig) -> str:
    if getattr(args, "input", None) and str(args.input).strip():
        return str(args.input).strip()
    if getattr(args, "aspects", None) and str(args.aspects).strip():
        return str(args.aspects).strip()
    if (cfg.input_tickets_csv or "").strip():
        return cfg.input_tickets_csv.strip()
    if (cfg.aspects_csv or "").strip():
        return cfg.aspects_csv.strip()
    return ""


def maybe_merge_aspects_file(df: pd.DataFrame, aspects_path: str) -> pd.DataFrame:
    """
    Merge aspect columns from a separate aspects CSV/XLSX onto the ticket dataframe.
    Preferred join: `idx` (if present in both and looks like a 0-based sequence).
    Fallback join: (language, subject, body) by exact match.
    """
    ap = (aspects_path or "").strip()
    if not ap:
        return df
    if not Path(ap).exists():
        raise FileNotFoundError(f"Aspects file not found: {ap}")

    asp = load_csv_or_excel(ap).fillna("")
    if asp.empty:
        log.warning(f"[Aspects] Aspects file is empty: {ap}")
        return df

    present = [c for c in ASPECT_COLS if c in asp.columns]
    if not present:
        log.warning(f"[Aspects] No aspect columns found in aspects file: {ap}")
        return df

    df = df.copy()

    # ── Preferred: join on idx (only if both have idx and it looks like 0-based) ──
    if "idx" in df.columns and "idx" in asp.columns:
        left_idx = pd.to_numeric(df["idx"], errors="coerce")
        right_idx = pd.to_numeric(asp["idx"], errors="coerce")

        left_ok = left_idx.notna().all()
        right_ok = right_idx.notna().all()

        # "starts from 0" check (0-based). We don't enforce strict contiguity,
        # but we do require min==0 and uniqueness to avoid bad merges.
        left_zero_based = left_ok and int(left_idx.min()) == 0
        right_zero_based = right_ok and int(right_idx.min()) == 0
        left_unique = left_ok and left_idx.is_unique
        right_unique = right_ok and right_idx.is_unique

        if left_zero_based and right_zero_based and left_unique and right_unique:
            left = df.copy()
            right = asp[["idx"] + present].copy()
            # normalize dtype to int for merge safety
            left["idx"] = pd.to_numeric(left["idx"], errors="raise").astype(int)
            right["idx"] = pd.to_numeric(right["idx"], errors="raise").astype(int)
            merged = left.merge(right, on="idx", how="left", suffixes=("", "_asp"))
            for c in present:
                if f"{c}_asp" in merged.columns:
                    merged[c] = merged.get(c, "").astype(str)
                    merged[f"{c}_asp"] = merged[f"{c}_asp"].astype(str)
                    merged[c] = merged[c].where(
                        merged[c].str.strip().ne(""),
                        merged[f"{c}_asp"],
                    )
                    merged = merged.drop(columns=[f"{c}_asp"])
            log.info(f"[Aspects] Merged aspects from {ap} by idx (0-based)")
            return merged
        else:
            log.info(
                "[Aspects] idx present but not usable for merge "
                f"(left_zero_based={left_zero_based}, right_zero_based={right_zero_based}, "
                f"left_unique={left_unique}, right_unique={right_unique}) — falling back to text keys"
            )

    # ── Fallback: exact match on subject/body (+ language if present) ──
    key_cols = [c for c in ["language", "subject", "body"] if c in df.columns and c in asp.columns]
    if "subject" not in key_cols or "body" not in key_cols:
        log.warning(
            f"[Aspects] Cannot merge aspects from {ap}: need at least subject+body columns in both files"
        )
        return df

    right = asp[key_cols + present].drop_duplicates(subset=key_cols, keep="last")
    merged = df.merge(right, on=key_cols, how="left", suffixes=("", "_asp"))
    for c in present:
        if f"{c}_asp" in merged.columns:
            merged[c] = merged.get(c, "").astype(str)
            merged[f"{c}_asp"] = merged[f"{c}_asp"].astype(str)
            merged[c] = merged[c].where(merged[c].str.strip().ne(""), merged[f"{c}_asp"])
            merged = merged.drop(columns=[f"{c}_asp"])
    log.info(f"[Aspects] Merged aspects from {ap} by {key_cols}")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model(config: GenerationConfig):
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               AutoModelForSeq2SeqLM, BitsAndBytesConfig)

    token    = get_hf_token()
    seq2seq  = any(config.model_id.lower().startswith(p)
                   for p in ("t5", "flan", "bart", "mt5"))
    ModelCls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM

    log.info(f"[Generator] Loading {config.model_id}  4bit={config.use_4bit}")

    if config.use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit              = True,
            bnb_4bit_compute_dtype    = torch.bfloat16,  # bfloat16 for gemma-2
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type       = "nf4",
        )
        model = ModelCls.from_pretrained(
            config.model_id, quantization_config=bnb,
            device_map="auto", token=token,
        )
    else:
        dtype = {"bfloat16": torch.bfloat16,
                 "float16" : torch.float16,
                 "float32" : torch.float32}.get(config.torch_dtype, torch.float16)
        model = ModelCls.from_pretrained(
            config.model_id, torch_dtype=dtype,
            device_map="auto", token=token,
        )

    tok = AutoTokenizer.from_pretrained(config.model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model.eval()
    log.info(f"[Generator] Ready | seq2seq={seq2seq}")
    return tok, model, seq2seq


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def build_baseline_prompt(row: pd.Series, body_max_chars: int = 0) -> str:
    """
    Zero-shot prompt for gemma-2-2b-it (instruction-tuned).
    Constrained to produce a concise, direct support reply — not a long article.
    gemma-2-2b-it is verbose by default; explicit length/format constraints are needed.
    """
    subject   = str(row.get("subject", "") or "")
    body_full = str(row.get("body",    "") or "")
    body      = body_full[:body_max_chars] if body_max_chars > 0 else body_full
    return (
        "Write a concise, professional customer support reply to the ticket below. "
        "Keep the reply focused and under 150 words. "
        "Do not use markdown headers or bullet points. "
        "Do not invent case studies or external links.\n\n"
        f"Subject: {subject}\n"
        f"Message: {body}\n\n"
        "Support reply:"
    )


def build_baseline_prompt_pt(row: pd.Series) -> str:
    """
    Zero-shot prompt for PRE-TRAINED models (no instruction tuning).
    Framed as text completion — no "Write a..." instruction.
    The model continues from "Support agent reply:" naturally.
    """
    subject = str(row.get("subject", "") or "")
    body    = str(row.get("body",    "") or "")
    return (
        f"Customer support ticket:\n"
        f"Subject: {subject}\n"
        f"Message: {body}\n\n"
        f"Support agent reply:\n"
    )


def build_aspect_only_prompt(row: pd.Series, spans: Dict,
                              body_max_chars: int = 0) -> str:
    """
    Aspect-only prompt — all 5 aspect columns injected into prompt slots.
    No retrieved context. Isolates the contribution of aspect extraction alone.
    """
    subject = str(row.get("subject", "") or "")
    body    = str(row.get("body", "") or "")
    lang    = str(row.get("language", "en") or "en")
    sentiment, severity, sent_compound = extract_sentiment_severity(body, lang)

    asp_str = " | ".join(
        f"{k}: {v}" for k, v in spans.items()
        if v and v not in ("none", "")
    ) or "GENERAL"

    prob    = spans.get("prob_sub", "") or spans.get("prob_statement", "")
    cause   = spans.get("cause", "")
    enforce = (
        f"The response MUST directly address: {prob}. "
        + (f"Root cause: {cause}. "
           if cause and cause not in ("none", "") else "")
    )
    # New fields from upgraded aspect extractor
    entities     = spans.get("entities",     "") or ""
    action_taken = spans.get("action_taken", "") or ""
    urgency_span = spans.get("urgency",      "") or ""

    # Build optional new-field lines
    extra = ""
    if entities:
        extra += f"Entities   : {entities}\n"
    if action_taken:
        extra += f"ActionTaken: {action_taken}\n"
    # Urgency from aspect extractor overrides rule-based severity if present
    urgency_line = urgency_span if urgency_span else severity

    return (
        "Write a concise, professional customer support reply. "
        "Keep the reply under 150 words. "
        "Do not use markdown headers or bullet points. "
        "Do not invent case studies or external links.\n"
        f"{enforce}\n\n"
        f"Aspects  : {asp_str}\n"
        f"{extra}"
        f"Sentiment: {sentiment} (compound={sent_compound:.3f})\n"
        f"Urgency  : {urgency_line}\n"
        f"Subject  : {subject}\n"
        f"Message  : {body}\n\n"
        "Support reply:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _apply_chat_template(tok, prompts: List[str], seq2seq: bool) -> List[str]:
    """
    Apply chat template to a list of prompts (causal LM only).
    gemma-2-2b-it has no chat template — try/except falls back to raw prompt.
    This is correct: PT model continues text naturally from the prompt ending.
    """
    if seq2seq:
        return prompts
    out = []
    for p in prompts:
        try:
            p = tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            pass   # PT model: no chat template, raw prompt used as-is
        out.append(p)
    return out


def generate_batch(prompts: List[str], tok, model, config: GenerationConfig,
                   seq2seq: bool) -> List[str]:
    """
    Batched generation — tokenize all prompts together, single model.generate() call.
    Left-pads so all sequences are aligned on the right (required for causal LMs).
    """
    prompts = _apply_chat_template(tok, prompts, seq2seq)

    dev = config.device
    try:
        dev = str(next(model.parameters()).device)
    except Exception:
        pass

    # Left-padding for causal LM batch decoding
    orig_padding_side = tok.padding_side
    tok.padding_side = "left"

    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,   # 2048 safe for gemma-2 — 4096 risks CUDA OOM at batch>1
    ).to(dev)

    tok.padding_side = orig_padding_side

    gen_kw = dict(
        max_new_tokens      = config.max_new_tokens,
        do_sample           = config.do_sample,
        pad_token_id        = tok.eos_token_id,
        repetition_penalty  = 1.3,   # PT models loop without this — penalises
                                     # repeated token sequences during generation
        no_repeat_ngram_size= 4,     # hard-blocks any 4-gram from repeating
    )
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)

    results = []
    for i, seq in enumerate(out):
        if seq2seq:
            results.append(tok.decode(seq, skip_special_tokens=True).strip())
        else:
            new_tokens = seq[inputs["input_ids"].shape[1]:]
            results.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def generate_one(prompt: str, tok, model, config: GenerationConfig,
                 seq2seq: bool) -> str:
    """Single-prompt wrapper around generate_batch (kept for compatibility)."""
    return generate_batch([prompt], tok, model, config, seq2seq)[0]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _get_spans(row: pd.Series, df: pd.DataFrame) -> Dict:
    """Extract non-empty aspect spans from a dataframe row."""
    spans = {}
    for k in ASPECT_COLS:
        if k in df.columns:
            v = str(row.get(k, "") or "")
            if v.lower() not in ("", "none", "nan"):
                spans[k] = v
    return spans


def _aspect_row(spans: Dict) -> Dict:
    """Return all 5 aspect values as a flat dict for storing in output rows."""
    return {k: spans.get(k, "") for k in ASPECT_COLS}


def extract_sentiment_severity(body: str, language: str) -> tuple[str, str, float]:
    """
    Returns (sentiment_label, severity_label, vader_compound).
    Sentiment uses VADER if available; otherwise returns neutral/0.
    Severity is a simple rule-based heuristic (no extra model required).
    """
    text = (body or "").strip()

    compound = 0.0
    if _VADER is not None and text:
        # VADER is English-optimized; still provides a useful signal on many tickets.
        compound = float(_VADER.polarity_scores(text).get("compound", 0.0))

    if compound >= 0.25:
        sentiment = "positive"
    elif compound <= -0.25:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    t = text.lower()
    # Strong outage/blocked-work signals
    critical_kw = [
        "down", "outage", "offline", "unavailable", "cannot access", "can't access",
        "cannot login", "can't login", "unable to login", "locked out", "security breach",
        "data loss", "lost data", "incident", "urgent", "asap", "immediately",
    ]
    high_kw = [
        "error", "failed", "failure", "not working", "doesn't work", "does not work",
        "crash", "crashing", "timeout", "time out", "bug", "broken",
        "payment", "charged", "refund", "invoice", "billing",
    ]
    medium_kw = [
        "slow", "latency", "performance", "delay",
        "integration", "api", "sync", "export", "import",
        "feature request", "request", "how do i", "how to",
    ]

    if any(k in t for k in critical_kw):
        severity = "critical"
    elif any(k in t for k in high_kw):
        severity = "high"
    elif any(k in t for k in medium_kw):
        severity = "medium"
    else:
        severity = "low"

    # Escalate if very negative sentiment and not already critical
    if compound <= -0.6 and severity in ("low", "medium"):
        severity = "high"

    return sentiment, severity, compound


def inject_sentiment_severity(prompt: str, sentiment: str, severity: str, compound: float) -> str:
    """
    Insert sentiment/severity lines near the top of the prompt.
    Keeps the rest of the prompt unchanged to avoid breaking existing formatting.
    """
    insert = f"Sentiment: {sentiment} (compound={compound:.3f})\nSeverity : {severity}\n"
    if not prompt:
        return insert
    # Place after the first line if possible.
    i = prompt.find("\n")
    if i == -1:
        return prompt + "\n" + insert
    return prompt[: i + 1] + insert + prompt[i + 1 :]


def _make_retriever(config: GenerationConfig):
    """Load FAISS retriever from config."""
    from dataclasses import replace
    from src.retrieval.retriever import TicketRetriever
    from src.retrieval.inference_retriever import load_retriever_config
    ret_cfg = replace(
        load_retriever_config(config.retrieval_config),
        top_k=config.top_k,
    )
    return TicketRetriever(ret_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# RUNNERS — one per ablation system
# ─────────────────────────────────────────────────────────────────────────────

def run_baseline(df: pd.DataFrame, tok, model, seq2seq: bool,
                 config: GenerationConfig) -> pd.DataFrame:
    """
    System 1 — Zero-shot baseline.
    No RAG, no aspects. Measures pure model capability.
    """
    ckpt     = Checkpointer("baseline", every=config.checkpoint_every)
    done     = ckpt.load()
    done_idx = {r["idx"] for r in done}
    rows     = list(done)
    t0       = time.time()
    bs       = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[Baseline] {len(pending)} rows to generate  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]
        prompts = [build_baseline_prompt(row, config.body_max_chars) for _, row in batch]
        responses = generate_batch(prompts, tok, model, config, seq2seq)

        for (i, row), prompt, response in zip(batch, prompts, responses):
            rows.append({
                "idx"              : int(row.get("idx", i)),
                "language"         : row.get("language", "en"),
                "subject"          : str(row.get("subject", "") or ""),
                "body"             : str(row.get("body",    "") or ""),
                "answer"           : str(row.get("answer",  "") or ""),
                "generated_answer" : response,
                "baseline_prompt"  : prompt,
                "baseline_response": response,
                "baseline_length"  : len(response.split()),
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[Baseline] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


def run_rag_only(df: pd.DataFrame, tok, model, seq2seq: bool,
                 config: GenerationConfig) -> pd.DataFrame:
    """
    System 2 — RAG only (no aspects).
    Dense E5+FAISS retrieval. No aspect spans passed to retriever or prompt.
    Isolates the contribution of retrieval alone vs baseline.
    """
    retriever = _make_retriever(config)
    ckpt      = Checkpointer("rag_only", every=config.checkpoint_every)
    done      = ckpt.load()
    done_idx  = {r["idx"] for r in done}
    rows      = list(done)
    t0        = time.time()
    bs        = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[RAG-only] {len(pending)} rows to generate  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]

        # retrieval is per-row (FAISS lookup) — build prompts first
        meta_batch, prompts = [], []
        for i, row in batch:
            subject = str(row.get("subject", "") or "")
            body    = str(row.get("body",    "") or "")
            lang    = str(row.get("language", "en") or "en")
            retrieved = retriever.retrieve(subject=subject, body=body,
                                           language=lang, aspect_spans=None)
            prompt = retriever.build_prompt(subject, body, retrieved, None)
            sent, sev, comp = extract_sentiment_severity(body, lang)
            prompt = inject_sentiment_severity(prompt, sent, sev, comp)
            prompts.append(prompt)
            meta_batch.append((i, row, subject, lang, sent, sev, len(retrieved), prompt))

        responses = generate_batch(prompts, tok, model, config, seq2seq)

        for (i, row, subject, lang, sent, sev, n_ret, prompt), response in zip(meta_batch, responses):
            rows.append({
                "idx"              : int(row.get("idx", i)),
                "language"         : lang,
                "subject"          : subject,
                "answer"           : str(row.get("answer", "") or ""),
                "generated_answer" : response,
                "sentiment"        : sent,
                "severity"         : sev,
                "rag_only_prompt"  : prompt,
                "rag_only_response": response,
                "rag_only_length"  : len(response.split()),
                "n_retrieved"      : n_ret,
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[RAG-only] {n_done}/{len(pending)}  {(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


def run_aspect_only(df: pd.DataFrame, tok, model, seq2seq: bool,
                    config: GenerationConfig) -> pd.DataFrame:
    """
    System 3 — Aspects only (no RAG).
    All 5 aspect columns from aspect_extractor injected into prompt slots:
      prob_sub, prob_statement, cause, intent, ticket_type_nli
    No FAISS retrieval. Isolates the contribution of aspect extraction alone.
    """
    ckpt     = Checkpointer("aspect_only", every=config.checkpoint_every)
    done     = ckpt.load()
    done_idx = {r["idx"] for r in done}
    rows     = list(done)
    t0       = time.time()
    bs       = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[Aspect-only] {len(pending)} rows to generate  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]

        meta_batch, prompts = [], []
        for i, row in batch:
            spans  = _get_spans(row, df)
            prompt = build_aspect_only_prompt(row, spans, config.body_max_chars)
            prompts.append(prompt)
            meta_batch.append((i, row, spans, prompt))

        responses = generate_batch(prompts, tok, model, config, seq2seq)

        for (i, row, spans, prompt), response in zip(meta_batch, responses):
            rows.append({
                "idx"                 : int(row.get("idx", i)),
                "language"            : row.get("language", "en"),
                "subject"             : str(row.get("subject", "") or ""),
                "answer"              : str(row.get("answer", "") or ""),
                "generated_answer"    : response,
                "aspect_only_prompt"  : prompt,
                "aspect_only_response": response,
                "aspect_only_length"  : len(response.split()),
                **_aspect_row(spans),
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[Aspect-only] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


def run_rag_aspect(df: pd.DataFrame, tok, model, seq2seq: bool,
                   config: GenerationConfig) -> pd.DataFrame:
    """
    System 4 — RAG + Aspects (full system).
    Aspects drive THREE things:
      (a) aspect_query_vector — weighted query embedding (subject+body+spans) for FAISS
      (b) cross-encoder reranking for EN / pure dense for DE
      (c) prompt slots — all 8 aspect fields in build_prompt()
    All 8 aspect columns (5 original + entities, action_taken, urgency) stored in output.
    """
    retriever = _make_retriever(config)
    ckpt      = Checkpointer("rag_aspect", every=config.checkpoint_every)
    done      = ckpt.load()
    done_idx  = {r["idx"] for r in done}
    rows      = list(done)
    t0        = time.time()
    bs        = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[RAG+Aspects] {len(pending)} rows to generate  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]

        meta_batch, prompts = [], []
        for i, row in batch:
            spans   = _get_spans(row, df)
            subject = str(row.get("subject", "") or "")
            body    = str(row.get("body",    "") or "")
            lang    = str(row.get("language", "en") or "en")
            retrieved = retriever.retrieve(
                subject=subject, body=body, language=lang,
                aspect_spans=(spans if spans else None),
            )
            prompt = retriever.build_prompt(
                subject, body, retrieved, (spans if spans else None))
            # NOTE: Sentiment/Severity NOT injected here for System 4.
            prompts.append(prompt)
            meta_batch.append((i, row, spans, subject, lang, len(retrieved), prompt))

        responses = generate_batch(prompts, tok, model, config, seq2seq)

        for (i, row, spans, subject, lang, n_ret, prompt), response in zip(meta_batch, responses):
            rows.append({
                "idx"                : int(row.get("idx", i)),
                "language"           : lang,
                "subject"            : subject,
                "answer"             : str(row.get("answer", "") or ""),
                "generated_answer"   : response,
                "rag_aspect_prompt"  : prompt,
                "rag_aspect_response": response,
                "rag_aspect_length"  : len(response.split()),
                "n_retrieved"        : n_ret,
                **_aspect_row(spans),
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[RAG+Aspects] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRA MODEL LOADER (T5-Base / Gemma-PT)
# ─────────────────────────────────────────────────────────────────────────────

def load_extra_model(model_key: str):
    """
    Load one of the extra baseline models from EXTRA_MODEL_REGISTRY.
    Returns (tokenizer, model, seq2seq, max_new_tokens).
    """
    from transformers import (AutoTokenizer, AutoModelForCausalLM,
                               AutoModelForSeq2SeqLM, BitsAndBytesConfig)

    spec     = EXTRA_MODEL_REGISTRY[model_key]
    model_id = spec["model_id"]
    seq2seq  = spec["seq2seq"]
    use_4bit = spec["use_4bit"]
    dtype_str= spec["torch_dtype"]
    ModelCls = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
    token    = get_hf_token()

    log.info(f"[{spec['label']}] Loading {model_id}  4bit={use_4bit}")
    if use_4bit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = ModelCls.from_pretrained(
            model_id, quantization_config=bnb,
            device_map="auto", token=token)
    else:
        dtype = {"bfloat16": torch.bfloat16,
                 "float16" : torch.float16,
                 "float32" : torch.float32}.get(dtype_str, torch.float32)
        model = ModelCls.from_pretrained(
            model_id, torch_dtype=dtype,
            device_map="auto", token=token)

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    log.info(f"[{spec['label']}] Ready | seq2seq={seq2seq}")
    return tok, model, seq2seq, spec["max_new_tokens"]


def generate_batch_extra(prompts: List[str], tok, model, seq2seq: bool,
                          max_new_tokens: int, do_sample: bool = False) -> List[str]:
    """
    Batched generation for extra models.
    Handles PT models (no chat template) and seq2seq (T5).
    max_length=2048 safe for quantized Gemma-PT with batch>1.
    """
    if not seq2seq:
        formatted = []
        for p in prompts:
            try:
                p = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True)
            except Exception:
                pass   # PT model — no chat template, use raw prompt
            formatted.append(p)
        prompts = formatted

    dev = "cuda"
    try:
        dev = str(next(model.parameters()).device)
    except Exception:
        pass

    orig = tok.padding_side
    tok.padding_side = "left"
    inputs = tok(prompts, return_tensors="pt", padding=True,
                 truncation=True, max_length=2048).to(dev)
    tok.padding_side = orig

    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                  pad_token_id=tok.eos_token_id,
                  repetition_penalty=1.3,
                  no_repeat_ngram_size=4)
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kw)

    results = []
    for i, seq in enumerate(out):
        if seq2seq:
            results.append(tok.decode(seq, skip_special_tokens=True).strip())
        else:
            new_tokens = seq[inputs["input_ids"].shape[1]:]
            results.append(tok.decode(new_tokens, skip_special_tokens=True).strip())
    return results


def run_extra_baseline(model_key: str, df: pd.DataFrame,
                        config: GenerationConfig) -> pd.DataFrame:
    """
    Runner for extra baseline models (T5-Base / Gemma-PT).
    Zero-shot only — no RAG, no aspects.
    PT-aware prompt used for gemma_pt.
    """
    spec      = EXTRA_MODEL_REGISTRY[model_key]
    label     = spec["label"]
    is_pt     = spec.get("is_pretrained", False)
    tok, model, seq2seq, max_new_tokens = load_extra_model(model_key)
    bs        = config.generation_batch_size

    ckpt     = Checkpointer(model_key, every=config.checkpoint_every)
    done     = ckpt.load()
    done_idx = {r["idx"] for r in done}
    rows     = list(done)
    t0       = time.time()

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[{label}] {len(pending)} rows  batch_size={bs}  pt={is_pt}")

    for b_start in range(0, len(pending), bs):
        batch   = pending[b_start : b_start + bs]
        # PT model uses completion-style prompt; IT model uses instruction prompt
        prompts = [
            build_baseline_prompt_pt(row) if is_pt
            else build_baseline_prompt(row)
            for _, row in batch
        ]
        responses = generate_batch_extra(
            prompts, tok, model, seq2seq,
            max_new_tokens=max_new_tokens,
            do_sample=config.do_sample,
        )
        for (i, row), prompt, response in zip(batch, prompts, responses):
            rows.append({
                "idx"                  : int(row.get("idx", i)),
                "language"             : row.get("language", "en"),
                "subject"              : str(row.get("subject", "") or ""),
                "answer"               : str(row.get("answer",  "") or ""),
                "generated_answer"     : response,
                f"{model_key}_prompt"  : prompt,
                f"{model_key}_response": response,
                f"{model_key}_length"  : len(response.split()),
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[{label}] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    del model, tok
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# GUIDED PROMPT BUILDER (from experiment_rag_prompt.py)
# ─────────────────────────────────────────────────────────────────────────────

def build_guided_rag_prompt(subject: str, body: str,
                             retrieved: List[Dict],
                             aspect_spans: Dict = None) -> str:
    """
    Guided RAG prompt — explicitly instructs the model to USE retrieved examples.
    Key improvement over plain context dump in build_prompt().
    Aspect slots included when aspect_spans provided.
    """
    spans = aspect_spans or {}

    core_asp = {k: v for k, v in spans.items()
                if k in ("prob_sub","prob_statement","cause",
                          "intent","ticket_type_nli")
                and v and v not in ("none", "")}
    asp_str      = " | ".join(f"{k}: {v}" for k, v in core_asp.items())
    entities     = spans.get("entities",     "") or ""
    action_taken = spans.get("action_taken", "") or ""
    urgency      = spans.get("urgency",      "") or ""
    prob  = spans.get("prob_sub", "") or spans.get("prob_statement", "")
    cause = spans.get("cause", "")

    ctx_parts = []
    for i, r in enumerate(retrieved):
        ctx_parts.append(
            f"[Example {i+1}]\n"
            f"  Subject : {r['subject']}\n"
            f"  Response: {r['response']}"
        )
    ctx_str = "\n\n".join(ctx_parts) if ctx_parts else "(no examples retrieved)"

    aspect_block = ""
    if asp_str:
        aspect_block += f"Aspects    : {asp_str}\n"
    if entities:
        aspect_block += f"Entities   : {entities}\n"
    if action_taken:
        aspect_block += f"ActionTaken: {action_taken}\n"
    if urgency:
        aspect_block += f"Urgency    : {urgency}\n"

    enforce = ""
    if prob:
        enforce = f"Your response MUST directly address: {prob}."
        if cause and cause not in ("none", ""):
            enforce += f" Root cause noted: {cause}."
        enforce += "\n"

    # IT model: concise instruction framing with format constraints
    return (
        f"You are a professional customer support agent. "
        f"Write a concise reply (under 150 words) to the new ticket below. "
        f"Use the similar resolved tickets as reference only. "
        f"Do not use markdown headers or bullet points. "
        f"Do not invent case studies or external links.\n\n"
        f"SIMILAR RESOLVED TICKETS:\n{ctx_str}\n\n"
        f"---\n"
        f"NEW TICKET:\n"
        f"Subject: {subject}\n"
        + (f"Issue: {prob}\n" if prob else "")
        + (f"Root cause: {cause}\n" if cause and cause not in ("none","") else "")
        + f"{aspect_block}"
        + f"Message: {body}\n\n"
        "Support reply:"
    )


def run_rag_guided(df: pd.DataFrame, tok, model, seq2seq: bool,
                   config: GenerationConfig) -> pd.DataFrame:
    """
    Guided RAG — pure dense retrieval + explicit use-context prompt.
    No aspects. Tests whether guided prompting improves over plain RAG.
    """
    retriever = _make_retriever(config)
    ckpt      = Checkpointer("rag_guided", every=config.checkpoint_every)
    done      = ckpt.load()
    done_idx  = {r["idx"] for r in done}
    rows      = list(done)
    t0        = time.time()
    bs        = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[RAG-Guided] {len(pending)} rows  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]
        meta_batch, prompts = [], []
        for i, row in batch:
            subject   = str(row.get("subject", "") or "")
            body      = str(row.get("body",    "") or "")
            lang      = str(row.get("language", "en") or "en")
            retrieved = retriever.retrieve(subject=subject, body=body,
                                           language=lang, aspect_spans=None)
            prompt    = build_guided_rag_prompt(subject, body, retrieved, None)
            prompts.append(prompt)
            meta_batch.append((i, row, subject, lang, len(retrieved), prompt))

        responses = generate_batch(prompts, tok, model, config, seq2seq)
        for (i, row, subject, lang, n_ret, prompt), response in                 zip(meta_batch, responses):
            rows.append({
                "idx"                : int(row.get("idx", i)),
                "language"           : lang,
                "subject"            : subject,
                "answer"             : str(row.get("answer", "") or ""),
                "generated_answer"   : response,
                "rag_guided_prompt"  : prompt,
                "rag_guided_response": response,
                "rag_guided_length"  : len(response.split()),
                "n_retrieved"        : n_ret,
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[RAG-Guided] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


def run_rag_asp_guided(df: pd.DataFrame, tok, model, seq2seq: bool,
                        config: GenerationConfig) -> pd.DataFrame:
    """
    Guided RAG + Aspects — aspect-aware retrieval + explicit use-context prompt.
    All 8 aspect fields injected. Tests guided prompting with full aspect info.
    """
    retriever = _make_retriever(config)
    ckpt      = Checkpointer("rag_asp_guided", every=config.checkpoint_every)
    done      = ckpt.load()
    done_idx  = {r["idx"] for r in done}
    rows      = list(done)
    t0        = time.time()
    bs        = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[RAG+Asp-Guided] {len(pending)} rows  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch = pending[b_start : b_start + bs]
        meta_batch, prompts = [], []
        for i, row in batch:
            spans     = _get_spans(row, df)
            subject   = str(row.get("subject", "") or "")
            body      = str(row.get("body",    "") or "")
            lang      = str(row.get("language", "en") or "en")
            retrieved = retriever.retrieve(
                subject=subject, body=body, language=lang,
                aspect_spans=(spans if spans else None))
            prompt = build_guided_rag_prompt(
                subject, body, retrieved,
                aspect_spans=(spans if spans else None))
            prompts.append(prompt)
            meta_batch.append((i, row, spans, subject, lang,
                               len(retrieved), prompt))

        responses = generate_batch(prompts, tok, model, config, seq2seq)
        for (i, row, spans, subject, lang, n_ret, prompt), response in                 zip(meta_batch, responses):
            rows.append({
                "idx"                    : int(row.get("idx", i)),
                "language"               : lang,
                "subject"                : subject,
                "answer"                 : str(row.get("answer", "") or ""),
                "generated_answer"       : response,
                "rag_asp_guided_prompt"  : prompt,
                "rag_asp_guided_response": response,
                "rag_asp_guided_length"  : len(response.split()),
                "n_retrieved"            : n_ret,
                **_aspect_row(spans),
            })
        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            log.info(f"[RAG+Asp-Guided] {n_done}/{len(pending)}  "
                     f"{(time.time()-t0)/60:.1f}m")

    ckpt.clear()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, prefix: str,
                    config: GenerationConfig) -> Dict:
    """
    Compute ROUGE, BLEU, BERTScore.
    prefix must match the response column: e.g. "rag_aspect" → "rag_aspect_response"
    """
    from dataclasses import dataclass as dc

    @dc
    class MetCfg:
        rouge_use_stemmer: bool = True
        rouge_types      : list = None
        bleu_tokenizer   : str  = "13a"
        bertscore_model  : str  = config.bertscore_model
        bertscore_batch  : int  = config.bertscore_batch
        bertscore_device : str  = config.bertscore_device
        def __post_init__(self):
            if self.rouge_types is None:
                self.rouge_types = ["rouge1", "rouge2", "rougeL"]

    calc     = MetricsCalculator(MetCfg())
    resp_col = f"{prefix}_response"

    if resp_col not in df.columns:
        log.warning(f"[Metrics] Missing column: {resp_col}")
        return {}
    if "answer" not in df.columns or df["answer"].fillna("").eq("").all():
        log.warning("[Metrics] No reference `answer` column — skipping metrics")
        return {}

    # Sort by idx before metric computation — prevents positional mismatch
    # if checkpoint resume changed row order vs original df.
    df_sorted = df.sort_values("idx").reset_index(drop=True)         if "idx" in df.columns else df
    hyps = df_sorted[resp_col].fillna("").tolist()
    refs = df_sorted["answer"].fillna("").tolist()

    results = {}
    results.update(calc.rouge(hyps, refs))
    results.update(calc.bleu(hyps, refs))
    results.update(calc.bertscore(hyps, refs))

    if "language" in df_sorted.columns:
        for lang in df_sorted["language"].unique():
            mask  = (df_sorted["language"] == lang).values
            h_sub = [h for h, m in zip(hyps, mask) if m]
            r_sub = [r for r, m in zip(refs,  mask) if m]
            if len(h_sub) >= 2:
                for k, v in {**calc.rouge(h_sub, r_sub),
                             **calc.bleu(h_sub, r_sub),
                             **calc.bertscore(h_sub, r_sub)}.items():
                    results[f"{k}_{lang}"] = v
    return results


def print_metrics(label: str, metrics: Dict):
    print(f"\n{'═'*65}\n  {label}\n{'═'*65}")
    for k in ["rouge1", "rouge2", "rougeL", "bleu1", "bleu4", "bertscore_f1"]:
        if k in metrics:
            print(f"    {k:<22}: {metrics[k]}")
    langs = sorted(set(k.split("_")[-1] for k in metrics
                       if k.startswith("rouge1_")))
    for lang in langs:
        print(f"  LANGUAGE: {lang.upper()}")
        for base in ["rouge1", "rouge2", "rougeL", "bleu1", "bleu4", "bertscore_f1"]:
            key = f"{base}_{lang}"
            if key in metrics:
                print(f"    {key:<28}: {metrics[key]}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

# Mode groups — used by run() orchestrator
_ORIGINAL_MODES  = {"baseline", "rag", "aspect", "rag_aspect"}
_EXTRA_MODES     = {"t5_base", "gemma_pt"}
_GUIDED_MODES    = {"rag_guided", "rag_asp_guided"}
_ALL_GROUP       = _ORIGINAL_MODES
_ALL_BASELINES   = {"baseline", "t5_base", "gemma_pt"}
_ALL_GUIDED      = _GUIDED_MODES
_ALL_MODES       = _ORIGINAL_MODES | _EXTRA_MODES | _GUIDED_MODES


def run(data_path: str, mode: str, config: GenerationConfig,
        n: Optional[int] = None,
        start_idx: Optional[int] = None,
        end_idx: Optional[int] = None):
    """
    Unified orchestrator — all modes:
      Original  : baseline | rag | aspect | rag_aspect | all
      Baselines : t5_base | gemma_pt | all_baselines
      Guided    : rag_guided | rag_asp_guided | all_guided
      Everything: all_modes
    Index range: start_idx..end_idx inclusive (supports 2-8 record smoke tests)
    """
    log.info(f"[Generation] mode={mode}  model={config.model_id}  input={data_path}")

    if not data_path:
        raise ValueError(
            "No input file. Set input_tickets_csv in generation_config.yaml "
            "or pass --input.")

    df = load_csv_or_excel(data_path)
    if "idx" not in df.columns:
        df["idx"] = df.index
    # Optional range filter by idx (inclusive).
    if start_idx is not None or end_idx is not None:
        idx_num = pd.to_numeric(df["idx"], errors="coerce")
        if idx_num.isna().any():
            raise ValueError("idx must be numeric to use --start_idx/--end_idx")
        s = int(start_idx) if start_idx is not None else int(idx_num.min())
        e = int(end_idx) if end_idx is not None else int(idx_num.max())
        df = df[idx_num.between(s, e)].reset_index(drop=True)
        log.info(f"[Generation] idx range filter: {s}–{e}  => {len(df)} rows")

    if n:
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
    log.info(f"[Generation] {len(df)} rows loaded from {data_path}")

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    base = re.sub(r"\.(csv|xlsx|xls)$", "", Path(data_path).name, flags=re.I)

    # ── ASPECT EXTRACTION ─────────────────────────────────────────────────
    # Required for system 3 (aspect) and system 4 (rag_aspect).
    # Saved to disk immediately after extraction to survive crashes.
    needs_aspects = mode in ("aspect", "rag_aspect", "all",
                             "rag_asp_guided", "all_guided", "all_modes")
    if needs_aspects:
        # Include idx range in cache filename to prevent stale cache
        # from a different run range being loaded silently.
        _range = (f"_{start_idx}_{end_idx}"
                  if (start_idx is not None or end_idx is not None)
                  else "_full")
        range_tag   = _range
        aspects_out = Path(config.output_dir) / f"{base}_with_aspects{range_tag}.csv"
        if aspects_out.exists() and not config.force_reextract_aspects:
            log.info(f"[Aspects] Loading saved aspects from {aspects_out}")
            df = pd.read_csv(aspects_out)
            # Reapply idx range filter — saved aspects file may contain more
            # rows than the current run requested (e.g. previous run was 1000
            # rows, current run is 200). Without this the filter is lost.
            if start_idx is not None or end_idx is not None:
                idx_num = pd.to_numeric(df["idx"], errors="coerce")
                s = int(start_idx) if start_idx is not None else int(idx_num.min())
                e = int(end_idx)   if end_idx   is not None else int(idx_num.max())
                df = df[idx_num.between(s, e)].reset_index(drop=True)
                log.info(f"[Aspects] Re-filtered to idx {s}–{e} => {len(df)} rows")
        else:
            # If the user provided a separate aspects file for this ticket set,
            # merge it first (then only extract if still missing or forced).
            if (config.aspects_file or "").strip():
                df = maybe_merge_aspects_file(df, config.aspects_file)
            df = enrich_dataframe_with_aspects(df, config)
            df.to_csv(aspects_out, index=False)
            log.info(f"[Aspects] Saved → {aspects_out}")

    completed   = {}
    metric_rows = []

    with Profiler() as prof:
        tok, model, seq2seq = load_model(config)

        # ── SYSTEM 1: BASELINE ─────────────────────────────────────────────
        if mode in ("baseline", "all"):
            log.info("[Generation] ── System 1: Baseline (zero-shot) ──")
            df_base = run_baseline(df, tok, model, seq2seq, config)
            out     = Path(config.output_dir) / f"{base}_baseline.csv"
            df_base.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_base, "baseline", config)
            print_metrics(f"{config.model_id} — 1. Baseline (zero-shot)", m)
            metric_rows.append({"system": "1_baseline", **m})
            completed["1_baseline"] = df_base

        # ── SYSTEM 2: RAG ONLY ─────────────────────────────────────────────
        if mode in ("rag", "all"):
            log.info("[Generation] ── System 2: RAG only (no aspects) ──")
            df_rag = run_rag_only(df, tok, model, seq2seq, config)
            out    = Path(config.output_dir) / f"{base}_rag_only.csv"
            df_rag.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_rag, "rag_only", config)
            print_metrics(f"{config.model_id} — 2. RAG only (no aspects)", m)
            metric_rows.append({"system": "2_rag_only", **m})
            completed["2_rag_only"] = df_rag

        # ── SYSTEM 3: ASPECTS ONLY ─────────────────────────────────────────
        if mode in ("aspect", "all"):
            log.info("[Generation] ── System 3: Aspects only (no RAG) ──")
            df_asp = run_aspect_only(df, tok, model, seq2seq, config)
            out    = Path(config.output_dir) / f"{base}_aspect_only.csv"
            df_asp.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_asp, "aspect_only", config)
            print_metrics(f"{config.model_id} — 3. Aspects only (no RAG)", m)
            metric_rows.append({"system": "3_aspect_only", **m})
            completed["3_aspect_only"] = df_asp

        # ── SYSTEM 4: RAG + ASPECTS (FULL SYSTEM) ─────────────────────────
        if mode in ("rag_aspect", "all"):
            log.info("[Generation] ── System 4: RAG + Aspects (full system) ──")
            df_ra = run_rag_aspect(df, tok, model, seq2seq, config)
            out   = Path(config.output_dir) / f"{base}_rag_aspect.csv"
            df_ra.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_ra, "rag_aspect", config)
            print_metrics(f"{config.model_id} — 4. RAG + Aspects (full system)", m)
            metric_rows.append({"system": "4_rag_aspect", **m})
            completed["4_rag_aspect"] = df_ra

        # ── SYSTEM 5: T5-BASE ZERO-SHOT ───────────────────────────────────
        if mode in ("t5_base", "all_baselines", "all_modes"):
            log.info("[Generation] ── System 5: T5-Base Zero-Shot ──")
            df_t5 = run_extra_baseline("t5_base", df, config)
            out   = Path(config.output_dir) / f"{base}_t5_base.csv"
            df_t5.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_t5, "t5_base", config)
            print_metrics("T5-Base Zero-Shot", m)
            metric_rows.append({"system": "5_t5_base", **m})
            completed["5_t5_base"] = df_t5

        # ── SYSTEM 6: GEMMA-3-1B PRE-TRAINED ──────────────────────────────
        if mode in ("gemma_pt", "all_baselines", "all_modes"):
            log.info("[Generation] ── System 6: Gemma-3-1B Pre-Trained ──")
            df_gpt = run_extra_baseline("gemma_pt", df, config)
            out    = Path(config.output_dir) / f"{base}_gemma_pt.csv"
            df_gpt.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_gpt, "gemma_pt", config)
            print_metrics("Gemma-3-1B Pre-Trained", m)
            metric_rows.append({"system": "6_gemma_pt", **m})
            completed["6_gemma_pt"] = df_gpt

        # ── SYSTEM 7: RAG GUIDED ──────────────────────────────────────────
        if mode in ("rag_guided", "all_guided", "all_modes"):
            log.info("[Generation] ── System 7: RAG + Guided Prompt ──")
            df_rg = run_rag_guided(df, tok, model, seq2seq, config)
            out   = Path(config.output_dir) / f"{base}_rag_guided.csv"
            df_rg.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_rg, "rag_guided", config)
            print_metrics("RAG + Guided Prompt", m)
            metric_rows.append({"system": "7_rag_guided", **m})
            completed["7_rag_guided"] = df_rg

        # ── SYSTEM 8: RAG + ASPECTS + GUIDED ─────────────────────────────
        if mode in ("rag_asp_guided", "all_guided", "all_modes"):
            log.info("[Generation] ── System 8: RAG + Aspects + Guided Prompt ──")
            df_rag = run_rag_asp_guided(df, tok, model, seq2seq, config)
            out    = Path(config.output_dir) / f"{base}_rag_asp_guided.csv"
            df_rag.to_csv(out, index=False)
            log.info(f"[Generation] Saved → {out}")
            m = compute_metrics(df_rag, "rag_asp_guided", config)
            print_metrics("RAG + Aspects + Guided Prompt", m)
            metric_rows.append({"system": "8_rag_asp_guided", **m})
            completed["8_rag_asp_guided"] = df_rag

    # ── COMPARISON TABLE ──────────────────────────────────────────────────────
    if len(completed) > 1:
        comp = compare_models(completed)
        out  = Path(config.output_dir) / f"{base}_comparison.csv"
        comp.to_csv(out, index=False)
        log.info(f"[Generation] Comparison saved → {out}")

    # ── METRICS SUMMARY ────────────────────────────────────────────────────
    if metric_rows:
        mdf = pd.DataFrame(metric_rows)
        out = Path(config.output_dir) / f"{base}_metrics.csv"
        mdf.to_csv(out, index=False)
        print(f"\n{'═'*65}\n  SUMMARY — 4-WAY ABLATION\n{'═'*65}")
        print(mdf.to_string(index=False))

    print(f"\n{prof.result.summary()}")
    log.info("[Generation] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=(
            "4-way ablation: baseline | rag | aspect | rag_aspect | all\n"
            "Input: --input <csv> or set input_tickets_csv in generation_config.yaml"
        ),
    )
    p.add_argument("--input",   default=None,
                   help="CSV/XLSX: subject, body, language, (optional) answer")
    p.add_argument("--aspects", default=None,
                   help="Pre-extracted aspects CSV (skips extractor if aspect cols present)")
    p.add_argument(
        "--aspects_file",
        default=None,
        help=(
            "Separate aspects CSV/XLSX to merge onto --input by idx "
            "(or exact match on language+subject+body). "
            "Useful for running aspects_only with two files."
        ),
    )
    p.add_argument(
        "--mode",
        default="all",
        choices=[
            # Original 4-way
            "baseline", "rag", "aspect", "aspects_only", "rag_aspect", "all",
            # Extra baselines
            "t5_base", "gemma_pt", "all_baselines",
            # Guided prompt experiment
            "rag_guided", "rag_asp_guided", "all_guided",
            # Everything
            "all_modes",
        ],
        help=(
            "baseline      — Gemma zero-shot, no RAG, no aspects\n"
            "rag           — Gemma RAG only (dense retrieval)\n"
            "aspect        — Gemma aspects only (no RAG)\n"
            "rag_aspect    — Gemma RAG + aspects (full system)\n"
            "all           — all four above (default)\n"
            "t5_base       — T5-Base zero-shot\n"
            "gemma_pt      — Gemma-3-1B pre-trained zero-shot\n"
            "all_baselines — baseline + t5_base + gemma_pt\n"
            "rag_guided    — RAG + guided use-context prompt\n"
            "rag_asp_guided— RAG + aspects + guided prompt\n"
            "all_guided    — rag_guided + rag_asp_guided\n"
            "all_modes     — all 8 systems"
        ),
    )
    p.add_argument("--config", default="generation_config.yaml")
    p.add_argument("--n", type=int, default=None,
                   help="rows to process (default: all)")
    p.add_argument("--start_idx", type=int, default=None,
                   help="process only rows with idx >= start_idx (inclusive)")
    p.add_argument("--end_idx", type=int, default=None,
                   help="process only rows with idx <= end_idx (inclusive)")
    args = p.parse_args()

    cfg = (load_generation_config(args.config)
           if Path(args.config).exists()
           else GenerationConfig())

    # Allow CLI override for separate aspects file merge
    if getattr(args, "aspects_file", None) and str(args.aspects_file).strip():
        cfg.aspects_file = str(args.aspects_file).strip()

    # Aliases
    if args.mode == "aspects_only":
        args.mode = "aspect"
    # Expand group modes to constituent modes for needs_aspects check
    # (run() uses string matching internally, so pass as-is)

    path = resolve_data_path(args, cfg)
    if not path:
        p.error(
            "No input file. Use --input <tickets.csv> or set "
            "input_tickets_csv in generation_config.yaml"
        )
    if not Path(path).exists():
        p.error(f"Input file not found: {path}")

    run(
        data_path=path,
        mode=args.mode,
        config=cfg,
        n=args.n,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
    )
