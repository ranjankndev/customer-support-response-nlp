"""
response_generator.py — T5-based Response Generation for Customer Support
==========================================================================

PIPELINE POSITION:
  aspects_*.csv  →  rag.py (build_index + retrieve + build_prompt)
                 →  response_generator.py (T5 generate + coverage check)
                 →  responses_*.csv

TWO MODES:
  1. BASELINE   — T5 on raw subject+body only (no aspects, no RAG context)
  2. RAG+ASPECT — T5 on structured aspect-slot prompt from rag.py

SKIPPING QA FINE-TUNING:
  Use --model google/flan-t5-base (zero-shot, no fine-tuning needed)
  After fine-tuning step3, swap to finetuned_t5/ for better results

USAGE:
  # Baseline only (fast, no RAG index needed)
  python response_generator.py --aspects aspects_1k.csv --mode baseline

  # RAG+Aspect (recommended)
  python response_generator.py --aspects aspects_1k.csv --mode rag

  # Both modes side by side (for comparison)
  python response_generator.py --aspects aspects_1k.csv --mode both

  # Limit rows for quick test
  python response_generator.py --aspects aspects_1k.csv --mode both --n 100

OUTPUTS:
  responses_{name}_baseline.csv   — raw T5 output on body only
  responses_{name}_rag.csv        — T5 output on RAG+aspect prompt
  responses_{name}_comparison.csv — both side by side + metrics delta

METRICS (no reference needed — all automatic):
  ROUGE-1 / ROUGE-2 / ROUGE-L   — n-gram overlap vs actual answer
  BERTScore F1                   — semantic similarity vs actual answer
  entity_coverage                — % of prob_sub/version/error in response
  avg_length                     — response word count
"""

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
# CONFIG — change these at the top, nowhere else
# ═══════════════════════════════════════════════════════════════════════════════

T5_MODEL        = 'google/flan-t5-large'   # swap to 'finetuned_t5/' after training
BODY_LEN        = 800                     # max chars from body in baseline prompt
MAX_NEW_TOKENS  = 200                     # max tokens T5 generates per response
NUM_BEAMS       = 4                       # beam search width
KB_SIZE         = None                    # None = use full dataset as KB; or int e.g. 5000

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
# T5 MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_t5(model_path: str = T5_MODEL):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    log.info(f"Loading T5: {model_path}")
    tok   = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = model.to(device)
    model.eval()
    log.info(f"  T5 ready on {device}")
    return tok, model, device


def generate_response(prompt: str, tok, model, device: str) -> str:
    import torch
    inputs = tok(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024,   # RAG prompt with body+context can exceed 512
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


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITY COVERAGE CHECK  [OUR]
# Checks how many key entities from prob_sub/prob_statement appear in response
# ═══════════════════════════════════════════════════════════════════════════════

_VERSION_RE = re.compile(r'\bv?\d+\.\d+[\.\d]*\b', re.I)
_ERROR_RE   = re.compile(r'\b[A-Z][A-Z0-9_]{2,}(?:Error|Exception|Fault|Code|ID)\b')

def check_entity_coverage(prob_sub: str, prob_statement: str, response: str) -> dict:
    """
    Extract key entities from aspect spans and check if response mentions them.
    Returns coverage score 0.0–1.0 and list of missing entities.
    """
    combined = f"{prob_sub} {prob_statement}"
    versions = _VERSION_RE.findall(combined)
    errors   = _ERROR_RE.findall(combined)

    # Core noun from prob_sub (first 2 tokens, skip generic words)
    _SKIP = {'the','a','an','is','are','was','were','our','your','this','that',
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
        'entity_coverage': round(coverage, 3),
        'missing_entities': ', '.join(missing) if missing else '',
    }


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_rouge(predictions: list, references: list) -> dict:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
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
        log.warning("rouge_score not installed — pip install rouge-score")
        return {'ROUGE-1': None, 'ROUGE-2': None, 'ROUGE-L': None}
    
def compute_bertscore(predictions: list, references: list) -> float:
    try:
        if not predictions or not references:
            return None
        from bert_score import score as bscore
        _, _, F = bscore(
            predictions, references,
            model_type='distilbert-base-uncased',  # ← lighter, faster
            verbose=False
        )
        return round(float(F.mean()), 4)
    except ImportError:
        log.warning("bert_score not installed — pip install bert-score")
        return None    


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE MODE — T5 on raw subject + body (no aspects, no RAG)
# ═══════════════════════════════════════════════════════════════════════════════

def build_baseline_prompt(row: pd.Series) -> str:
    subject = str(row.get('subject', '') or '')
    body    = str(row.get('body',    '') or '')[:BODY_LEN]
    return (
        f"Write a professional customer support response to the following ticket.\n\n"
        f"Subject: {subject}\n"
        f"Complaint: {body}\n"
        f"Response:"
    )


def run_baseline(df: pd.DataFrame, tok, model, device: str) -> pd.DataFrame:
    log.info(f"[BASELINE] Generating {len(df)} responses...")
    t0 = time.time()
    rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        prompt   = build_baseline_prompt(row)
        response = generate_response(prompt, tok, model, device)

        cov = check_entity_coverage(
            str(row.get('prob_sub', '')       or ''),
            str(row.get('prob_statement', '') or ''),
            response,
        )
        rows.append({
            'idx':             row.get('idx', i),
            'language':        row.get('language', 'en'),
            'subject':         row.get('subject', ''),
            'actual_answer':   str(row.get('answer', '') or ''),
            'baseline_prompt': prompt,
            'baseline_response': response,
            'baseline_length': len(response.split()),
            **{f'baseline_{k}': v for k, v in cov.items()},
        })

        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(df)}]  {(time.time()-t0)/60:.1f}m elapsed")

    log.info(f"[BASELINE] Done in {(time.time()-t0)/60:.1f}m")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# RAG+ASPECT MODE — T5 on structured aspect-slot prompt from rag.py
# ═══════════════════════════════════════════════════════════════════════════════

def run_rag(df: pd.DataFrame, tok, model, device: str, kb_df: pd.DataFrame = None) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).parent))
    from rag import RAGSystem

    log.info("[RAG] Building index...")
    rag = RAGSystem()

    _kb = kb_df if kb_df is not None else (
        df.sample(min(KB_SIZE, len(df)), random_state=42) if KB_SIZE else df
    )
    log.info(f"[RAG] KB size: {len(_kb)} | Query size: {len(df)}")
    rag.build_index(_kb)

    log.info(f"[RAG] Generating {len(df)} responses...")
    t0   = time.time()
    rows = []

    for i, (_, row) in enumerate(df.iterrows()):
        lang    = str(row.get('language', 'en') or 'en').lower()[:2]
        subject = str(row.get('subject', '') or '')
        body    = str(row.get('body',    '') or '')

        # Build aspect spans from pre-computed aspect columns
        spans = {
            k: str(row.get(k, '') or '')
            for k in ['prob_sub', 'prob_statement', 'cause',
                      'categorization', 'priority', 'urgency_vibe']
            if str(row.get(k, '') or '') not in ('', 'none', 'nan')
        }

        retrieved = rag.retrieve(
            query        = f"{subject} {body}",
            aspect_spans = spans,
            subject      = subject,
            lang         = lang,
        )
        prompt   = rag.build_prompt(row, retrieved, spans)
        response = generate_response(prompt, tok, model, device)

        # [OUR] Level-2 aspect enforcement — verify coverage, retry if low
        cov = check_entity_coverage(
            spans.get('prob_sub', ''),
            spans.get('prob_statement', ''),
            response,
        )
        if cov['entity_coverage'] < 0.5 and spans.get('prob_sub', ''):
            prob  = spans.get('prob_sub', '')
            cause = spans.get('cause', '')
            stronger = prompt.replace(
                "Write a professional customer support response to the following ticket.",
                f"Write a detailed customer support response that MUST explicitly mention "
                f"'{prob}'"
                + (f" and address the root cause '{cause}'" if cause and cause != 'none' else "")
                + "."
            )
            response = generate_response(stronger, tok, model, device)
            cov = check_entity_coverage(
                spans.get('prob_sub', ''),
                spans.get('prob_statement', ''),
                response,
            )
            log.debug(f"[RAG] Row {i} — retried (coverage was low)")
        rows.append({
            'idx':          row.get('idx', i),
            'language':     row.get('language', 'en'),
            'subject':      subject,
            'actual_answer': str(row.get('answer', '') or ''),
            'rag_prompt':   prompt,
            'rag_response': response,
            'rag_length':   len(response.split()),
            'n_retrieved':  len(retrieved),
            **{f'rag_{k}': v for k, v in cov.items()},
        })

        if (i + 1) % 50 == 0:
            log.info(f"  [{i+1}/{len(df)}]  {(time.time()-t0)/60:.1f}m elapsed")

    log.info(f"[RAG] Done in {(time.time()-t0)/60:.1f}m")
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def print_metrics(label: str, df: pd.DataFrame, prefix: str):
    col_resp = f'{prefix}_response'
    col_ans  = 'actual_answer'

    if col_resp not in df.columns or col_ans not in df.columns:
        return

    preds = df[col_resp].fillna('').tolist()
    refs  = df[col_ans].fillna('').tolist()

    rouge  = compute_rouge(preds, refs)
    bscore = compute_bertscore(
        [p for p, r in zip(preds, refs) if r],
        [r for r in refs if r],
    )
    cov_col = f'{prefix}_entity_coverage'
    avg_cov = df[cov_col].mean() if cov_col in df.columns else None
    avg_len = df[f'{prefix}_length'].mean() if f'{prefix}_length' in df.columns else None

    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  ROUGE-1          : {rouge['ROUGE-1']}")
    print(f"  ROUGE-2          : {rouge['ROUGE-2']}")
    print(f"  ROUGE-L          : {rouge['ROUGE-L']}")
    print(f"  BERTScore F1     : {bscore}")
    if avg_cov is not None:
        print(f"  Entity Coverage  : {avg_cov:.3f}")
    if avg_len is not None:
        print(f"  Avg Length (words): {avg_len:.1f}")
    print(f"{'─'*50}")

    return {
        'mode': label,
        'ROUGE-1': rouge['ROUGE-1'],
        'ROUGE-2': rouge['ROUGE-2'],
        'ROUGE-L': rouge['ROUGE-L'],
        'BERTScore': bscore,
        'entity_coverage': round(avg_cov, 3) if avg_cov else None,
        'avg_length': round(avg_len, 1) if avg_len else None,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run(aspects_path: str,
        mode: str       = 'both',
        n: int          = None,
        model_path: str = T5_MODEL,
        kb_path: str    = None):
    """
    kb_path : path to full aspects CSV used as RAG knowledge base.
              If None, uses aspects_path — fine for small tests.
              For proper eval pass full 28k CSV here and use n= for query sample.
    """

    log.info(f"Loading aspects: {aspects_path}")
    df = pd.read_csv(aspects_path) if aspects_path.endswith('.csv') \
         else pd.read_excel(aspects_path)

    # body + answer must be present — aspect_pipeline.py now carries them
    if 'body' not in df.columns:
        log.warning("'body' column missing — RAG prompts will be empty. "
                    "Re-run aspect_pipeline.py (body+answer now included).")
    if 'answer' not in df.columns:
        log.warning("'answer' column missing — metrics vs reference will be skipped.")

    if n:
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)
        log.info(f"Sampled {len(df)} rows for query")

    # KB for RAG — use full dataset if kb_path provided
    if kb_path:
        log.info(f"Loading KB from: {kb_path}")
        kb_df = pd.read_csv(kb_path) if kb_path.endswith('.csv') \
                else pd.read_excel(kb_path)
        log.info(f"KB size: {len(kb_df)} rows")
    else:
        kb_df = df
        log.info(f"KB = query set ({len(kb_df)} rows) — pass kb_path= for proper eval")

    log.info(f"Dataset: {len(df)} rows | mode: {mode}")

    base      = re.sub(r'\.(csv|xlsx)$', '', Path(aspects_path).name)
    tok, model_t5, device = load_t5(model_path)
    metric_rows = []

    # ── Baseline ──────────────────────────────────────────────────────────────
    if mode in ('baseline', 'both'):
        df_base = run_baseline(df, tok, model_t5, device)
        out     = f"responses_{base}_baseline.csv"
        df_base.to_csv(out, index=False)
        log.info(f"Saved → {out}")
        m = print_metrics("BASELINE (no RAG, no aspects)", df_base, 'baseline')
        if m: metric_rows.append(m)

    # ── RAG+Aspect ────────────────────────────────────────────────────────────
    if mode in ('rag', 'both'):
        df_rag = run_rag(df, tok, model_t5, device, kb_df=kb_df)
        out    = f"responses_{base}_rag.csv"
        df_rag.to_csv(out, index=False)
        log.info(f"Saved → {out}")
        m = print_metrics("RAG + ASPECT", df_rag, 'rag')
        if m: metric_rows.append(m)

    # ── Side-by-side comparison ───────────────────────────────────────────────
    if mode == 'both':
        comp = df[['idx','language','subject','answer']].copy() \
               if 'answer' in df.columns \
               else df[['idx','language','subject']].copy()

        comp = comp.merge(
            df_base[['idx','baseline_response','baseline_length',
                     'baseline_entity_coverage']],
            on='idx', how='left'
        ).merge(
            df_rag[['idx','rag_response','rag_length','rag_entity_coverage']],
            on='idx', how='left'
        )
        out = f"responses_{base}_comparison.csv"
        comp.to_csv(out, index=False)
        log.info(f"Saved comparison → {out}")

    # ── Metrics summary table ─────────────────────────────────────────────────
    if metric_rows:
        print(f"\n{'═'*50}")
        print("  SUMMARY")
        print(f"{'═'*50}")
        metrics_df = pd.DataFrame(metric_rows)
        print(metrics_df.to_string(index=False))
        metrics_df.to_csv(f"metrics_{base}.csv", index=False)
        print(f"\n  Saved → metrics_{base}.csv")

    log.info("Done.")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='T5 response generation — baseline vs RAG+aspect'
    )
    p.add_argument('--aspects', required=True,
                   help='Path to aspects_*.csv from aspect_pipeline.py')
    p.add_argument('--mode', default='both',
                   choices=['baseline', 'rag', 'both'],
                   help='baseline | rag | both (default: both)')
    p.add_argument('--n', type=int, default=None,
                   help='Number of rows to process (default: all)')
    p.add_argument('--model', default=T5_MODEL,
                   help=f'T5 model path (default: {T5_MODEL})')
    p.add_argument('--kb', default=None,
                   help='Path to full aspects CSV used as RAG KB (default: same as --aspects)')
    args = p.parse_args()

    run(
        aspects_path = args.aspects,
        mode         = args.mode,
        n            = args.n,
        model_path   = args.model,
        kb_path      = args.kb,
    )
