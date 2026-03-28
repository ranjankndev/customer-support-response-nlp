"""
response_generator.py — T5-based Response Generation for Customer Support
==========================================================================

PIPELINE POSITION:
  aspects_*.csv  →  rag.py (build_index + retrieve + build_prompt)
                 →  response_generator.py (T5 generate + coverage check)
                 →  responses_*.csv

TWO MODES (original, unchanged):
  1. BASELINE   — T5 on raw subject+body only (no aspects, no RAG context)
  2. RAG+ASPECT — T5 on structured aspect-slot prompt from rag.py

ADDITIONAL BASELINE MODES (new):
  3. LLAMA      — LLaMA-3.2-1B-Instruct zero-shot (no RAG, no aspects)
  4. GEMMA      — Gemma-2-2B-IT zero-shot (no RAG, no aspects)

NOTE — LLaMA license:
  Accept at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
  Then set:  export HF_TOKEN=hf_...
  On Kaggle: Notebook Settings → Add Secret → HF_TOKEN

USAGE:
  # Quick end-to-end test (10 rows, T5 only — no HF token needed)
  python response_generator.py --aspects aspect_results_500.csv --mode both --n 10

  # All four baselines (accept LLaMA license first)
  python response_generator.py --aspects aspect_results_500.csv --mode all

  # Single instruct model
  python response_generator.py --aspects aspect_results_500.csv --mode llama

OUTPUTS:
  responses_{name}_baseline.csv
  responses_{name}_rag.csv
  responses_{name}_llama.csv
  responses_{name}_gemma.csv
  responses_{name}_comparison.csv   — all completed modes side-by-side
  metrics_{name}.csv                — ROUGE / BLEU / BERTScore summary table

CHECKPOINTING (Kaggle-safe):
  Progress saved every CHECKPOINT_EVERY rows → .ckpt_{name}_{mode}.csv
  Resumes automatically if the session is interrupted mid-run.

METRICS:
  ROUGE-1/2/L  (n-gram overlap vs actual_answer)
  BLEU-1/2/4   (corpus-level, SmoothingFunction.method1)
  BERTScore F1 (distilbert-base-uncased)
  entity_coverage  (key noun / version / error present in response)
  avg_length       (word count)
"""

import os
import re
import sys
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG — change at the top only
# ═══════════════════════════════════════════════════════════════════════════════

T5_MODEL         = 'google/flan-t5-large'
LLAMA_MODEL      = 'meta-llama/Llama-3.2-1B-Instruct'
GEMMA_MODEL      = 'google/gemma-2-2b-it'   # Gemma-2 2B IT — better than Gemma-1 1B, same VRAM

BODY_LEN         = 800     # max chars from body used in prompts
MAX_NEW_TOKENS   = 200     # tokens generated per response
NUM_BEAMS        = 4       # T5 beam search width
KB_SIZE          = None    # None = full dataset as KB; or int e.g. 5000
CHECKPOINT_EVERY = 50      # save progress every N rows (Kaggle session safety)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('response_gen')

# ═══════════════════════════════════════════════════════════════════════════════
# HF TOKEN HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_hf_token():
    """
    Read HF token from — in order:
      1. Environment variable HF_TOKEN or HUGGINGFACE_TOKEN
      2. Google Colab userdata secrets  (Colab: key icon → Add secret → HF_TOKEN)
      3. Kaggle UserSecretsClient       (Kaggle: Notebook Settings → Add Secret)
    Returns None if not found on any platform — never raises.
    """
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    if token:
        return token

    # Colab secrets — only attempt if running in Colab
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass

    # Kaggle secrets — only attempt if running in Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret('HF_TOKEN')
        if token:
            return token
    except Exception:
        pass

    return None


def _check_llama_token():
    """
    Warn clearly if HF_TOKEN is missing before attempting LLaMA download.
    Does NOT abort — lets the download attempt surface the real HuggingFace error.
    """
    if not _get_hf_token():
        log.warning('=' * 64)
        log.warning('  HF_TOKEN not found.  LLaMA 3.2 requires:')
        log.warning('  1. Accept the license at:')
        log.warning('     https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct')
        log.warning('  2. Set your token via ONE of:')
        log.warning('     • export HF_TOKEN=hf_...                (terminal)')
        log.warning('     • Colab: key icon (🔑) → Add secret → HF_TOKEN')
        log.warning('     • Kaggle: Notebook Settings → Add Secret → HF_TOKEN')
        log.warning('  Gemma and T5 modes do NOT need a token.')
        log.warning('=' * 64)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═══════════════════════════════════════════════════════════════════════════════

def _ckpt_path(base: str, mode: str) -> str:
    return f'.ckpt_{base}_{mode}.csv'


def _load_checkpoint(base: str, mode: str) -> list:
    path = _ckpt_path(base, mode)
    if Path(path).exists():
        existing = pd.read_csv(path)
        log.info(f'[RESUME] Checkpoint: {path} ({len(existing)} rows already done)')
        return existing.to_dict('records')
    return []


def _save_checkpoint(rows: list, base: str, mode: str):
    pd.DataFrame(rows).to_csv(_ckpt_path(base, mode), index=False)


def _clear_checkpoint(base: str, mode: str):
    path = _ckpt_path(base, mode)
    if Path(path).exists():
        os.remove(path)


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_t5(model_path: str = T5_MODEL):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    log.info(f'Loading T5: {model_path}')
    tok    = AutoTokenizer.from_pretrained(model_path)
    model  = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device).eval()
    log.info(f'  T5 ready on {device}')
    return tok, model, device


def load_instruct_model(model_id: str):
    """
    Generic loader for causal instruction-tuned models (LLaMA, Gemma).
    - GPU available (Colab T4/A100, Kaggle T4): float16 + device_map=auto
    - CPU only (Colab free tier without GPU): float32, no device_map
    Returns (tokenizer, model).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    token  = _get_hf_token()
    has_gpu = torch.cuda.is_available()
    log.info(f'Loading instruct model: {model_id}  '
             f'({"GPU float16" if has_gpu else "CPU float32 — will be slow"})')

    tok = AutoTokenizer.from_pretrained(model_id, token=token)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if has_gpu:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.float16,
            device_map  = 'auto',
            token       = token,
        )
    else:
        # CPU fallback — works but ~10x slower; fine for --n 5 smoke test
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype = torch.float32,
            token       = token,
        )

    model.eval()
    log.info(f'  {model_id.split("/")[-1]} ready')
    return tok, model


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_t5(prompt: str, tok, model, device: str) -> str:
    import torch
    inputs = tok(
        prompt, return_tensors='pt', truncation=True, max_length=1024
    ).to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens       = MAX_NEW_TOKENS,
            min_new_tokens       = 40,
            num_beams            = NUM_BEAMS,
            early_stopping       = True,
            no_repeat_ngram_size = 3,
            length_penalty       = 2.0,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


def _generate_instruct(prompt: str, tok, model) -> str:
    """
    Greedy decode for causal LMs.
    Uses chat template if available (LLaMA / Gemma both support it),
    falls back to plain prompt if not.
    """
    import torch

    try:
        messages = [
            {'role': 'system', 'content': 'You are a helpful customer support agent.'},
            {'role': 'user',   'content': prompt},
        ]
        formatted = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        formatted = prompt

    device = next(model.parameters()).device
    inputs = tok(
        formatted, return_tensors='pt', truncation=True, max_length=1024
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,
            pad_token_id   = tok.eos_token_id,
        )
    # Decode only newly generated tokens — strip the input prompt
    new_tokens = out[0][inputs['input_ids'].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY COVERAGE CHECK  [OUR] — unchanged from original
# ═══════════════════════════════════════════════════════════════════════════════

_VERSION_RE = re.compile(r'\bv?\d+\.\d+[\.\d]*\b', re.I)
_ERROR_RE   = re.compile(r'\b[A-Z][A-Z0-9_]{2,}(?:Error|Exception|Fault|Code|ID)\b')

def check_entity_coverage(prob_sub: str, prob_statement: str, response: str) -> dict:
    combined   = f'{prob_sub} {prob_statement}'
    versions   = _VERSION_RE.findall(combined)
    errors     = _ERROR_RE.findall(combined)
    _SKIP      = {'the','a','an','is','are','was','were','our','your','this','that',
                  'issue','problem','error','bug','request','question','ticket'}
    core_nouns = [
        w for w in prob_sub.lower().split()[:6]
        if len(w) > 3 and w not in _SKIP
    ][:2]
    must_mention = list(set(versions + errors + core_nouns))
    resp_lower   = response.lower()
    missing      = [e for e in must_mention if e.lower() not in resp_lower]
    coverage     = 1.0 - (len(missing) / len(must_mention)) if must_mention else 1.0
    return {
        'entity_coverage':  round(coverage, 3),
        'missing_entities': ', '.join(missing) if missing else '',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT BUILDER — shared by all three zero-shot baselines (fair comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_zero_shot_prompt(row: pd.Series) -> str:
    subject = str(row.get('subject', '') or '')
    body    = str(row.get('body',    '') or '')[:BODY_LEN]
    return (
        'Write a professional customer support response to the following ticket.\n\n'
        f'Subject: {subject}\n'
        f'Complaint: {body}\n'
        'Response:'
    )

# Backward-compatible alias
build_baseline_prompt = _build_zero_shot_prompt


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge(predictions: list, references: list) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
        )
        r1, r2, rl = [], [], []
        for pred, ref in zip(predictions, references):
            if not ref or not pred:
                continue
            s = scorer.score(ref, pred)
            r1.append(s['rouge1'].fmeasure)
            r2.append(s['rouge2'].fmeasure)
            rl.append(s['rougeL'].fmeasure)
        return {
            'ROUGE-1': round(float(np.mean(r1)), 4) if r1 else 0.0,
            'ROUGE-2': round(float(np.mean(r2)), 4) if r2 else 0.0,
            'ROUGE-L': round(float(np.mean(rl)), 4) if rl else 0.0,
        }
    except ImportError:
        log.warning('rouge_score not installed — pip install rouge-score')
        return {'ROUGE-1': None, 'ROUGE-2': None, 'ROUGE-L': None}


def compute_bleu(predictions: list, references: list) -> dict:
    """
    Corpus-level BLEU-1, BLEU-2, BLEU-4 with SmoothingFunction.method1.
    Report BLEU-2 in the main table (consistent depth with ROUGE-2).
    """
    try:
        import nltk
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)

        smoother = SmoothingFunction().method1
        pairs    = [(r, p) for r, p in zip(references, predictions) if r and p]
        if not pairs:
            return {'BLEU-1': 0.0, 'BLEU-2': 0.0, 'BLEU-4': 0.0}

        refs = [[r.split()] for r, _ in pairs]   # corpus_bleu: list-of-list-of-tokens
        hyps = [p.split()   for _, p in pairs]

        b1 = corpus_bleu(refs, hyps, weights=(1, 0, 0, 0),            smoothing_function=smoother)
        b2 = corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0),        smoothing_function=smoother)
        b4 = corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=smoother)
        return {
            'BLEU-1': round(float(b1), 4),
            'BLEU-2': round(float(b2), 4),
            'BLEU-4': round(float(b4), 4),
        }
    except ImportError:
        log.warning('nltk not installed — pip install nltk')
        return {'BLEU-1': None, 'BLEU-2': None, 'BLEU-4': None}


def compute_bertscore(predictions: list, references: list):
    try:
        if not predictions or not references:
            return None
        from bert_score import score as bscore
        _, _, F = bscore(
            predictions, references,
            model_type = 'distilbert-base-uncased',
            verbose    = False,
        )
        return round(float(F.mean()), 4)
    except ImportError:
        log.warning('bert_score not installed — pip install bert-score')
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC ZERO-SHOT RUNNER — shared by T5-baseline, LLaMA, Gemma
# ═══════════════════════════════════════════════════════════════════════════════

def _run_zero_shot(
    df:     pd.DataFrame,
    mode:   str,        # 'baseline' | 'llama' | 'gemma'
    tok,
    model,
    device: str,        # used for T5 only; instruct models use device_map='auto'
    base:   str,        # checkpoint base name (derived from CSV filename)
    is_t5:  bool = False,
) -> pd.DataFrame:
    """
    Shared runner for all zero-shot baselines.
    All three use the same _build_zero_shot_prompt() — ensures fair comparison.
    Saves checkpoint every CHECKPOINT_EVERY rows and resumes on restart.
    """
    done     = _load_checkpoint(base, mode)
    done_set = {r['idx'] for r in done}
    rows     = list(done)
    t0       = time.time()
    pending  = [(i, row) for i, row in df.iterrows()
                if int(row.get('idx', i)) not in done_set]

    log.info(f'[{mode.upper()}] Generating {len(pending)} responses '
             f'({len(done)} resumed from checkpoint)')

    for n_done, (i, row) in enumerate(pending):
        prompt   = _build_zero_shot_prompt(row)
        response = (
            _generate_t5(prompt, tok, model, device)
            if is_t5
            else _generate_instruct(prompt, tok, model)
        )

        cov = check_entity_coverage(
            str(row.get('prob_sub',       '') or ''),
            str(row.get('prob_statement', '') or ''),
            response,
        )
        rows.append({
            'idx':              int(row.get('idx', i)),
            'language':         row.get('language', 'en'),
            'subject':          row.get('subject', ''),
            'actual_answer':    str(row.get('answer', '') or ''),
            f'{mode}_prompt':   prompt,
            f'{mode}_response': response,
            f'{mode}_length':   len(response.split()),
            **{f'{mode}_{k}': v for k, v in cov.items()},
        })

        if (n_done + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(rows, base, mode)
            elapsed = time.time() - t0
            rem     = elapsed / (n_done + 1) * (len(pending) - n_done - 1)
            log.info(f'  [{n_done+1:>4}/{len(pending)}]  {elapsed/60:.1f}m  '
                     f'~{rem/60:.1f}m left')

    _clear_checkpoint(base, mode)
    log.info(f'[{mode.upper()}] Done in {(time.time()-t0)/60:.1f}m')
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC RUNNERS
# ═══════════════════════════════════════════════════════════════════════════════

# Original function signature kept intact for backward compatibility
def generate_response(prompt: str, tok, model, device: str) -> str:
    return _generate_t5(prompt, tok, model, device)


def run_baseline(df: pd.DataFrame, tok, model, device: str,
                 base: str = 'run') -> pd.DataFrame:
    """T5 zero-shot baseline — original behaviour, now with checkpointing."""
    return _run_zero_shot(df, 'baseline', tok, model, device, base, is_t5=True)


def run_llama(df: pd.DataFrame, base: str = 'run') -> pd.DataFrame:
    """
    LLaMA-3.2-1B-Instruct zero-shot.
    Requires HF_TOKEN + accepted license before calling.
    """
    _check_llama_token()
    tok, model = load_instruct_model(LLAMA_MODEL)
    result = _run_zero_shot(df, 'llama', tok, model, device=None, base=base, is_t5=False)
    del model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def run_gemma(df: pd.DataFrame, base: str = 'run') -> pd.DataFrame:
    """Gemma-2-2B-IT zero-shot. No HF token required."""
    tok, model = load_instruct_model(GEMMA_MODEL)
    result = _run_zero_shot(df, 'gemma', tok, model, device=None, base=base, is_t5=False)
    del model
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# RAG+ASPECT MODE — original logic, now with checkpointing
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag(df: pd.DataFrame, tok, model, device: str,
            kb_df: pd.DataFrame = None, base: str = 'run') -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).parent))
    from rag import RAGSystem

    log.info('[RAG] Building index...')
    rag = RAGSystem()
    _kb = kb_df if kb_df is not None else (
        df.sample(min(KB_SIZE, len(df)), random_state=42) if KB_SIZE else df
    )
    log.info(f'[RAG] KB size: {len(_kb)} | Query size: {len(df)}')
    rag.build_index(_kb)

    done     = _load_checkpoint(base, 'rag')
    done_set = {r['idx'] for r in done}
    rows     = list(done)
    t0       = time.time()
    pending  = [(i, row) for i, row in df.iterrows()
                if int(row.get('idx', i)) not in done_set]

    log.info(f'[RAG] Generating {len(pending)} responses '
             f'({len(done)} resumed from checkpoint)')

    for n_done, (i, row) in enumerate(pending):
        lang    = str(row.get('language', 'en') or 'en').lower()[:2]
        subject = str(row.get('subject', '') or '')
        body    = str(row.get('body',    '') or '')

        spans = {
            k: str(row.get(k, '') or '')
            for k in ['prob_sub', 'prob_statement', 'cause',
                      'categorization', 'priority', 'urgency_vibe']
            if str(row.get(k, '') or '') not in ('', 'none', 'nan')
        }

        retrieved = rag.retrieve(
            query        = f'{subject} {body}',
            aspect_spans = spans,
            subject      = subject,
            lang         = lang,
        )
        prompt   = rag.build_prompt(row, retrieved, spans)
        response = _generate_t5(prompt, tok, model, device)

        # Level-2 aspect enforcement — retry if entity coverage low [OUR]
        cov = check_entity_coverage(
            spans.get('prob_sub', ''),
            spans.get('prob_statement', ''),
            response,
        )
        if cov['entity_coverage'] < 0.5 and spans.get('prob_sub', ''):
            prob  = spans.get('prob_sub', '')
            cause = spans.get('cause', '')
            stronger = prompt.replace(
                'Write a professional customer support response to the following ticket.',
                f'Write a detailed customer support response that MUST explicitly mention '
                f'"{prob}"'
                + (f' and address the root cause "{cause}"'
                   if cause and cause != 'none' else '')
                + '.'
            )
            response = _generate_t5(stronger, tok, model, device)
            cov = check_entity_coverage(
                spans.get('prob_sub', ''),
                spans.get('prob_statement', ''),
                response,
            )

        rows.append({
            'idx':           int(row.get('idx', i)),
            'language':      row.get('language', 'en'),
            'subject':       subject,
            'actual_answer': str(row.get('answer', '') or ''),
            'rag_prompt':    prompt,
            'rag_response':  response,
            'rag_length':    len(response.split()),
            'n_retrieved':   len(retrieved),
            **{f'rag_{k}': v for k, v in cov.items()},
        })

        if (n_done + 1) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(rows, base, 'rag')
            elapsed = time.time() - t0
            rem     = elapsed / (n_done + 1) * (len(pending) - n_done - 1)
            log.info(f'  [{n_done+1:>4}/{len(pending)}]  {elapsed/60:.1f}m  '
                     f'~{rem/60:.1f}m left')

    _clear_checkpoint(base, 'rag')
    log.info(f'[RAG] Done in {(time.time()-t0)/60:.1f}m')
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS PRINT + COLLECT
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics(label: str, df: pd.DataFrame, prefix: str) -> dict:
    col_resp = f'{prefix}_response'
    col_ans  = 'actual_answer'

    if col_resp not in df.columns or col_ans not in df.columns:
        return None

    preds = df[col_resp].fillna('').tolist()
    refs  = df[col_ans].fillna('').tolist()

    # Only score rows that have a reference answer
    valid = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    if not valid:
        log.warning(f'[{label}] No reference answers found — metrics skipped')
        return None
    vp, vr = zip(*valid)

    rouge  = compute_rouge(list(vp), list(vr))
    bleu   = compute_bleu(list(vp), list(vr))
    bscore = compute_bertscore(list(vp), list(vr))

    cov_col = f'{prefix}_entity_coverage'
    avg_cov = df[cov_col].mean()            if cov_col            in df.columns else None
    avg_len = df[f'{prefix}_length'].mean() if f'{prefix}_length' in df.columns else None

    print(f"\n{'─'*56}")
    print(f"  {label}")
    print(f"{'─'*56}")
    print(f"  ROUGE-1           : {rouge.get('ROUGE-1')}")
    print(f"  ROUGE-2           : {rouge.get('ROUGE-2')}")
    print(f"  ROUGE-L           : {rouge.get('ROUGE-L')}")
    print(f"  BLEU-1            : {bleu.get('BLEU-1')}")
    print(f"  BLEU-2            : {bleu.get('BLEU-2')}")
    print(f"  BLEU-4            : {bleu.get('BLEU-4')}")
    print(f"  BERTScore F1      : {bscore}")
    if avg_cov is not None:
        print(f"  Entity Coverage   : {avg_cov:.3f}")
    if avg_len is not None:
        print(f"  Avg Length (words): {avg_len:.1f}")
    print(f"{'─'*56}")

    return {
        'mode':            label,
        'ROUGE-1':         rouge.get('ROUGE-1'),
        'ROUGE-2':         rouge.get('ROUGE-2'),
        'ROUGE-L':         rouge.get('ROUGE-L'),
        'BLEU-1':          bleu.get('BLEU-1'),
        'BLEU-2':          bleu.get('BLEU-2'),
        'BLEU-4':          bleu.get('BLEU-4'),
        'BERTScore':       bscore,
        'entity_coverage': round(avg_cov, 3) if avg_cov is not None else None,
        'avg_length':      round(avg_len, 1) if avg_len is not None else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(aspects_path: str,
        mode:         str = 'both',
        n:            int = None,
        model_path:   str = T5_MODEL,
        kb_path:      str = None):
    """
    mode choices:
      baseline — T5 zero-shot
      rag      — T5 RAG+Aspect
      both     — T5 baseline + T5 RAG  (original behaviour)
      llama    — LLaMA-3.2-1B-Instruct zero-shot
      gemma    — Gemma-2-2B-IT zero-shot
      all      — all four modes
    """
    log.info(f'Loading: {aspects_path}')
    df = (pd.read_csv(aspects_path) if aspects_path.endswith('.csv')
          else pd.read_excel(aspects_path))

    if 'body'   not in df.columns:
        log.warning("'body' column missing — RAG prompts will be empty.")
    if 'answer' not in df.columns:
        log.warning("'answer' column missing — metrics vs reference skipped.")

    if n:
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
        log.info(f'Sampled {len(df)} rows')

    # Ensure idx column exists (needed by checkpointing)
    if 'idx' not in df.columns:
        df['idx'] = df.index

    kb_df = df
    if kb_path:
        log.info(f'Loading KB: {kb_path}')
        kb_df = (pd.read_csv(kb_path) if kb_path.endswith('.csv')
                 else pd.read_excel(kb_path))
        log.info(f'KB: {len(kb_df)} rows')

    base        = re.sub(r'\.(csv|xlsx)$', '', Path(aspects_path).name)
    metric_rows = []
    completed   = {}   # mode → DataFrame

    # ── T5 modes — load model once, reuse for both baseline and rag ──────────
    need_t5 = mode in ('baseline', 'rag', 'both', 'all')
    if need_t5:
        tok_t5, model_t5, device_t5 = load_t5(model_path)

    if mode in ('baseline', 'both', 'all'):
        df_base = run_baseline(df, tok_t5, model_t5, device_t5, base=base)
        out     = f'responses_{base}_baseline.csv'
        df_base.to_csv(out, index=False)
        log.info(f'Saved → {out}')
        m = print_metrics('T5 zero-shot (baseline)', df_base, 'baseline')
        if m: metric_rows.append(m)
        completed['baseline'] = df_base

    if mode in ('rag', 'both', 'all'):
        df_rag = run_rag(df, tok_t5, model_t5, device_t5, kb_df=kb_df, base=base)
        out    = f'responses_{base}_rag.csv'
        df_rag.to_csv(out, index=False)
        log.info(f'Saved → {out}')
        m = print_metrics('T5 + RAG + Aspect (ours)', df_rag, 'rag')
        if m: metric_rows.append(m)
        completed['rag'] = df_rag

    # Free T5 VRAM before loading instruct models
    if need_t5 and mode in ('llama', 'gemma', 'all'):
        del model_t5
        try:
            import torch; torch.cuda.empty_cache()
        except Exception:
            pass

    if mode in ('llama', 'all'):
        df_llama = run_llama(df, base=base)
        out      = f'responses_{base}_llama.csv'
        df_llama.to_csv(out, index=False)
        log.info(f'Saved → {out}')
        m = print_metrics('LLaMA-3.2-1B zero-shot', df_llama, 'llama')
        if m: metric_rows.append(m)
        completed['llama'] = df_llama

    if mode in ('gemma', 'all'):
        df_gemma = run_gemma(df, base=base)
        out      = f'responses_{base}_gemma.csv'
        df_gemma.to_csv(out, index=False)
        log.info(f'Saved → {out}')
        m = print_metrics('Gemma-2-2B zero-shot', df_gemma, 'gemma')
        if m: metric_rows.append(m)
        completed['gemma'] = df_gemma

    # ── Side-by-side comparison CSV ──────────────────────────────────────────
    if len(completed) > 1:
        id_cols = ['idx', 'language', 'subject']
        if 'answer' in df.columns:
            id_cols.append('answer')
        comp = df[id_cols].copy()
        for m_mode, m_df in completed.items():
            merge_cols = ['idx'] + [
                c for c in [f'{m_mode}_response', f'{m_mode}_length',
                             f'{m_mode}_entity_coverage']
                if c in m_df.columns
            ]
            comp = comp.merge(m_df[merge_cols], on='idx', how='left')
        out = f'responses_{base}_comparison.csv'
        comp.to_csv(out, index=False)
        log.info(f'Saved comparison → {out}')

    # ── Metrics summary table ─────────────────────────────────────────────────
    if metric_rows:
        print(f"\n{'═'*56}")
        print('  SUMMARY')
        print(f"{'═'*56}")
        mdf = pd.DataFrame(metric_rows)
        print(mdf.to_string(index=False))
        out = f'metrics_{base}.csv'
        mdf.to_csv(out, index=False)
        print(f'\n  Saved → {out}')

    log.info('All done.')


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Response generation — T5 / LLaMA / Gemma baselines + RAG+Aspect'
    )
    p.add_argument('--aspects', required=True,
                   help='Path to aspects_*.csv from aspect_pipeline.py')
    p.add_argument('--mode', default='both',
                   choices=['baseline', 'rag', 'both', 'llama', 'gemma', 'all'],
                   help='baseline | rag | both | llama | gemma | all  (default: both)')
    p.add_argument('--n', type=int, default=None,
                   help='Rows to process (default: all 500)')
    p.add_argument('--model', default=T5_MODEL,
                   help=f'T5 model path (default: {T5_MODEL})')
    p.add_argument('--kb', default=None,
                   help='Full KB CSV for RAG (default: same as --aspects)')
    args = p.parse_args()

    run(
        aspects_path = args.aspects,
        mode         = args.mode,
        n            = args.n,
        model_path   = args.model,
        kb_path      = args.kb,
    )
