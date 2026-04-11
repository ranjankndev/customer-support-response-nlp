"""
baseline_t5.py — T5-Base Zero-Shot Baseline
=============================================================
Runs google-t5/t5-base in zero-shot mode on the support ticket
dataset and computes the same metric stack as response_generator_v2.py
for direct comparison.

MODEL:
  google-t5/t5-base — 220M parameter encoder-decoder (seq2seq)
  No fine-tuning, no instruction tuning, no RAG, no aspects.
  Pure zero-shot: T5 is given the ticket and asked to generate a response.

WHY T5 AS A BASELINE:
  T5-base is a seq2seq model pretrained on C4 with a text-to-text objective.
  It represents what a small non-autoregressive baseline can do without
  any retrieval or structured aspect injection. Comparing it against
  Gemma-3-1B-PT (autoregressive) and the RAG/aspect systems isolates the
  contribution of model architecture, retrieval, and aspect extraction.

PROMPT:
  T5 does not use chat templates. The prompt is a plain instruction:
    "Answer the following customer support ticket.\n\nSubject: ...\nMessage: ...\nResponse:"
  T5 reads the full encoder input and generates the decoder output.

SAME METRIC STACK as response_generator_v2.py:
  - ROUGE (1, 2, L) with Porter stemming
  - BLEU (1, 4) with 13a tokenizer
  - BERTScore F1 with bert-base-multilingual-cased
  - Per-language breakdown (en / de)
  - Sorted by idx before metric computation

SAME INFRA:
  - generation_config.yaml (metric settings reused; model_id ignored)
  - src.utils: Checkpointer, MetricsCalculator, log, load_csv_or_excel
  - Batched generation with left-padding
  - Checkpointing every N rows (checkpoint key: t5_base)
  - start_idx / end_idx range filter

USAGE:
  python -m src.generation.baseline_t5 \\
    --config config/generation_config.yaml \\
    --input  data/processed/cs-support-multi-test.csv \\
    --start_idx 0 --end_idx 199

  # Full run:
  python -m src.generation.baseline_t5 \\
    --config config/generation_config.yaml \\
    --input  data/processed/cs-support-multi-test.csv

OUTPUT:
  output/generation/{input_base}_t5_base.csv       — generated responses
  output/generation/{input_base}_t5_base_metrics.csv — metric summary
"""

import re
import time
import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import yaml

from src.utils import (
    get_hf_token,
    load_csv_or_excel,
    Checkpointer,
    MetricsCalculator,
    log,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_ID       = "google/flan-t5-base"
MAX_NEW_TOKENS = 256      # flan-t5-base generates short-medium outputs
CKPT_KEY       = "t5_base"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — reads metric settings from generation_config.yaml
# model_id in the YAML is ignored; T5 is always loaded
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class T5Config:
    device               : str  = "cuda"
    do_sample            : bool = False
    output_dir           : str  = "output/generation"
    checkpoint_every     : int  = 50
    generation_batch_size: int  = 64   # T5-base is small — 16 is safe
    bertscore_model      : str  = "bert-base-multilingual-cased"
    bertscore_device     : str  = "cuda"
    bertscore_batch      : int  = 32


def load_t5_config(path: str) -> T5Config:
    """Load metric + infra settings from generation_config.yaml."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    allowed = T5Config.__dataclass_fields__
    return T5Config(**{k: v for k, v in cfg.items() if k in allowed})


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    """
    Load google-t5/t5-base in float32.
    T5-base is only 220M parameters — float32 fits easily on any GPU.
    No quantization needed or beneficial for this size.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    token = get_hf_token()
    log.info(f"[T5-Base] Loading {MODEL_ID}")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
        token=token,
    )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    log.info(f"[T5-Base] Ready  |  params ~220M  |  dtype=float32")
    return tok, model


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(row: pd.Series) -> str:
    """
    Flan-T5 instruction prompt.

    WHY FLAN-T5 INSTEAD OF T5-BASE:
      t5-base is raw pretrained (span-denoising on C4) — it cannot follow
      instructions. Any prompt produces True/False (NLI) or echoes the input.

      flan-t5-base is the same architecture (250M params) but instruction-tuned
      on 1800+ tasks including dialogue, QA, and text generation. It follows
      natural language instructions reliably zero-shot.

    PROMPT FORMAT:
      Flan-T5 works best with direct imperative instructions.
      No special prefixes needed — plain English instruction is sufficient.
    """
    subject = str(row.get("subject", "") or "")
    body    = str(row.get("body",    "") or "")
    return (
        f"Write a professional customer support reply to the following ticket.\n\n"
        f"Subject: {subject}\n"
        f"Message: {body}\n\n"
        f"Reply:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# BATCHED GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_batch(prompts: List[str], tok, model,
                   config: T5Config) -> List[str]:
    """
    Batched seq2seq generation.
    Left-padding for uniform batch shapes.
    T5 does not use sliding-window KV cache — safe at higher batch sizes.
    max_length=512 for encoder input (T5-base default context window).
    """
    dev = "cuda"
    try:
        dev = str(next(model.parameters()).device)
    except Exception:
        pass

    orig = tok.padding_side
    tok.padding_side = "left"
    inputs = tok(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,    # T5-base encoder max length
    ).to(dev)
    tok.padding_side = orig

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = MAX_NEW_TOKENS,
            do_sample            = config.do_sample,
            pad_token_id         = tok.pad_token_id,
            repetition_penalty   = 1.3,
            no_repeat_ngram_size = 3,   # T5 uses 3 — shorter outputs than LLMs
        )

    # seq2seq: decode full output (no prompt slicing needed)
    return [tok.decode(seq, skip_special_tokens=True).strip()
            for seq in out]


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_t5_base(df: pd.DataFrame, tok, model,
                config: T5Config) -> pd.DataFrame:
    """
    Zero-shot T5-base generation with checkpointing.
    Response column: t5_base_response (matches compute_metrics prefix).
    """
    ckpt     = Checkpointer(CKPT_KEY, every=config.checkpoint_every)
    done     = ckpt.load()
    done_idx = {r["idx"] for r in done}
    rows     = list(done)
    t0       = time.time()
    bs       = config.generation_batch_size

    pending = [(i, row) for i, row in df.iterrows()
               if int(row.get("idx", i)) not in done_idx]
    log.info(f"[T5-Base] {len(pending)} rows to generate  batch_size={bs}")

    for b_start in range(0, len(pending), bs):
        batch   = pending[b_start : b_start + bs]
        prompts = [build_prompt(row) for _, row in batch]

        responses = generate_batch(prompts, tok, model, config)

        for (i, row), prompt, response in zip(batch, prompts, responses):
            rows.append({
                "idx"              : int(row.get("idx", i)),
                "language"         : str(row.get("language", "en") or "en"),
                "subject"          : str(row.get("subject", "") or ""),
                "answer"           : str(row.get("answer",  "") or ""),
                "generated_answer" : response,
                "t5_base_prompt"   : prompt,
                "t5_base_response" : response,
                "t5_base_length"   : len(response.split()),
            })

        n_done = min(b_start + bs, len(pending))
        ckpt.save_if_due(rows, n_done)
        if n_done % 100 == 0 or n_done == len(pending):
            elapsed = (time.time() - t0) / 60
            speed   = n_done / elapsed if elapsed > 0 else 0
            log.info(f"[T5-Base] {n_done}/{len(pending)}  "
                     f"{elapsed:.1f}m  ({speed:.0f} rows/min)")

    ckpt.clear()
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, config: T5Config) -> Dict:
    """
    ROUGE, BLEU, BERTScore — identical config to response_generator_v2.py.
    Sorts by idx before computation for consistent ordering.
    Per-language breakdown for EN and DE.
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

    if "t5_base_response" not in df.columns:
        log.warning("[Metrics] t5_base_response column missing")
        return {}
    if "answer" not in df.columns or df["answer"].fillna("").eq("").all():
        log.warning("[Metrics] No reference answer column — skipping metrics")
        return {}

    calc = MetricsCalculator(MetCfg())
    df_s = df.sort_values("idx").reset_index(drop=True) \
        if "idx" in df.columns else df

    hyps = df_s["t5_base_response"].fillna("").tolist()
    refs = df_s["answer"].fillna("").tolist()

    results = {}
    results.update(calc.rouge(hyps, refs))
    results.update(calc.bleu(hyps, refs))
    results.update(calc.bertscore(hyps, refs))

    if "language" in df_s.columns:
        for lang in df_s["language"].unique():
            mask  = (df_s["language"] == lang).values
            h_sub = [h for h, m in zip(hyps, mask) if m]
            r_sub = [r for r, m in zip(refs,  mask) if m]
            if len(h_sub) >= 2:
                for k, v in {**calc.rouge(h_sub, r_sub),
                             **calc.bleu(h_sub, r_sub),
                             **calc.bertscore(h_sub, r_sub)}.items():
                    results[f"{k}_{lang}"] = v
    return results


def print_metrics(metrics: Dict):
    print(f"\n{'═'*65}\n  T5-Base Zero-Shot\n{'═'*65}")
    for k in ["rouge1","rouge2","rougeL","bleu1","bleu4","bertscore_f1"]:
        if k in metrics:
            print(f"    {k:<22}: {metrics[k]:.4f}")
    langs = sorted(set(k.split("_")[-1] for k in metrics
                       if k.startswith("rouge1_")))
    for lang in langs:
        print(f"  [{lang.upper()}]")
        for base in ["rouge1","rouge2","rougeL","bleu1","bleu4","bertscore_f1"]:
            key = f"{base}_{lang}"
            if key in metrics:
                print(f"    {key:<28}: {metrics[key]:.4f}")
    print(f"{'═'*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(data_path: str, config: T5Config,
        start_idx: Optional[int] = None,
        end_idx  : Optional[int] = None):

    log.info(f"[T5-Base] input={data_path}  model={MODEL_ID}")

    df = load_csv_or_excel(data_path)
    if "idx" not in df.columns:
        df["idx"] = df.index

    if start_idx is not None or end_idx is not None:
        idx_num = pd.to_numeric(df["idx"], errors="coerce")
        s = int(start_idx) if start_idx is not None else int(idx_num.min())
        e = int(end_idx)   if end_idx   is not None else int(idx_num.max())
        df = df[idx_num.between(s, e)].reset_index(drop=True)
        log.info(f"[T5-Base] idx filter: {s}–{e}  => {len(df)} rows")

    log.info(f"[T5-Base] {len(df)} rows loaded")
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    base = re.sub(r"\.(csv|xlsx|xls)$", "", Path(data_path).name, flags=re.I)

    tok, model = load_model()

    df_out = run_t5_base(df, tok, model, config)

    # Save responses
    out_csv = Path(config.output_dir) / f"{base}_t5_base.csv"
    df_out.to_csv(out_csv, index=False)
    log.info(f"[T5-Base] Responses saved → {out_csv}")

    # Compute and save metrics
    m = compute_metrics(df_out, config)
    print_metrics(m)

    mdf = pd.DataFrame([{"system": "t5_base", "model": MODEL_ID, **m}])
    out_metrics = Path(config.output_dir) / f"{base}_t5_base_metrics.csv"
    mdf.to_csv(out_metrics, index=False)
    log.info(f"[T5-Base] Metrics saved → {out_metrics}")

    log.info("[T5-Base] Done.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="T5-Base zero-shot baseline for customer support response generation")
    p.add_argument("--config",    default="config/generation_config.yaml",
                   help="generation_config.yaml (metric settings reused; model_id ignored)")
    p.add_argument("--input",     required=True,
                   help="Input CSV: subject, body, language, answer")
    p.add_argument("--start_idx", type=int, default=None,
                   help="Process rows with idx >= start_idx (inclusive)")
    p.add_argument("--end_idx",   type=int, default=None,
                   help="Process rows with idx <= end_idx (inclusive)")
    args = p.parse_args()

    cfg = load_t5_config(args.config) if Path(args.config).exists() \
          else T5Config()

    run(
        data_path = args.input,
        config    = cfg,
        start_idx = args.start_idx,
        end_idx   = args.end_idx,
    )
