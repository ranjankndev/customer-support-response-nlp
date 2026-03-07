"""
sbert_comparison.py
===================
Pretrained vs fine-tuned SBERT for EN+DE support ticket retrieval.

Model : paraphrase-multilingual-mpnet-base-v2
        Single model handles both English and German tickets in one
        aligned embedding space — no language routing needed.

Fine-tuning:
  Loss     : MultipleNegativesRankingLoss (in-batch negatives)
  Pairs    : (query_text, answer_text) per ticket
  Strategy : body + subject + tags -> answer
  Why      : No explicit negative mining needed. In a batch of 16,
             each pair treats the other 15 answers as negatives.

Metrics:
  MRR@10            -- does the right answer rank #1?
  Hit@1             -- top-1 precision
  NDCG@10           -- full ranked list quality
  intra_cluster_sim -- do same-tag tickets cluster together?

Usage:
  pip install sentence-transformers scikit-learn
  python sbert_comparison.py --data customer_support_28k_fixed.csv
  python sbert_comparison.py --data customer_support_28k_fixed.csv --epochs 5 --batch_size 32
"""

import os
os.environ['WANDB_DISABLED'] = 'true'   # suppress W&B prompt on Kaggle

import argparse
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from sklearn.model_selection import train_test_split
    from datasets import Dataset
except ImportError:
    raise SystemExit("pip install sentence-transformers scikit-learn datasets")

BASE_MODEL = 'paraphrase-multilingual-mpnet-base-v2'

BODY_LEN   = 800   # max chars from ticket body fed to SBERT
ANSWER_LEN = 800   # max chars from answer fed to SBERT

# ── Mojibake fix (same as corpus_builder) ────────────────────────────────────
_MOJIBAKE = {
    '\u00c3\u00bc': 'ü', '\u00c3\u00b6': 'ö', '\u00c3\u00a4': 'ä',
    '\u00c3\u009f': 'ß', '\u00c3\u009c': 'Ü', '\u00c3\u0096': 'Ö',
    '\u00c3\u0084': 'Ä',
}
_MRE = re.compile('|'.join(re.escape(k) for k in sorted(_MOJIBAKE, key=len, reverse=True)))

def _fix_encoding(text):
    return _MRE.sub(lambda m: _MOJIBAKE[m.group(0)], str(text))


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_tickets(path: str, n: int = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_excel(path) if path.endswith(('.xlsx', '.xls')) else pd.read_csv(path)
    df = df.dropna(subset=['body', 'answer'])
    df = df[df['body'].str.len() > 20]
    df = df[df['answer'].str.len() > 20]

    # Fix encoding
    garbled = sum(1 for t in df['body'].fillna('') if '\u00c3' in str(t))
    if garbled > 0:
        print(f"[Data] Fixing encoding in {garbled} rows...")
        for col in ['body', 'subject', 'answer']:
            if col in df.columns:
                df[col] = df[col].fillna('').apply(
                    lambda x: _fix_encoding(x) if '\u00c3' in str(x) else str(x)
                )

    tag_cols = [c for c in df.columns if c.startswith('tag_')]

    def build_query(row):
        tags = ' '.join(
            str(row[t]).replace(',', ' ')
            for t in tag_cols
            if pd.notna(row[t]) and str(row[t]) not in ('nan', '')
        )
        subj = str(row.get('subject', '') or '')
        body = str(row.get('body', '') or '')[:BODY_LEN]
        return f"{subj} {body} {tags}".strip()

    df['query_text']  = df.apply(build_query, axis=1)
    df['answer_text'] = df['answer'].astype(str).str[:ANSWER_LEN]

    if n and n < len(df):
        df = df.sample(n, random_state=seed).reset_index(drop=True)

    lang_dist = df['language'].value_counts().to_dict() if 'language' in df.columns else {}
    print(f"[Data] {len(df)} tickets | {lang_dist}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a @ b.T


def retrieval_metrics(q_vecs: np.ndarray, kb_vecs: np.ndarray, k: int = 10) -> dict:
    """
    Self-retrieval: query[i] should retrieve kb[i] at rank 1.
    Masks the diagonal to avoid trivial self-match.
    """
    sim = cosine_sim(q_vecs, kb_vecs)
    mrr, hit, ndcg = [], [], []

    for i in range(len(q_vecs)):
        scores = sim[i].copy()
        scores[i] = -999.0
        rank = int(np.sum(sim[i] > sim[i, i]))
        mrr.append(1.0 / (rank + 1) if rank < k else 0.0)
        hit.append(1.0 if rank == 0 else 0.0)
        ndcg.append((1.0 / np.log2(rank + 2)) if rank < k else 0.0)

    return {
        'MRR@10':  round(float(np.mean(mrr)),  4),
        'Hit@1':   round(float(np.mean(hit)),  4),
        'NDCG@10': round(float(np.mean(ndcg)), 4),
    }


def cluster_sim(vecs: np.ndarray, labels: list) -> float:
    """Mean intra-cluster cosine similarity — higher = better tag clustering."""
    sims = []
    for lbl in set(labels):
        idx = [i for i, l in enumerate(labels) if l == lbl]
        if len(idx) < 2:
            continue
        cv  = vecs[idx]
        sm  = cosine_sim(cv, cv)
        n   = len(idx)
        sims.append(float(np.mean(sm[np.triu_indices(n, k=1)])))
    return round(float(np.mean(sims)) if sims else 0.0, 4)


# ─────────────────────────────────────────────────────────────────────────────
# FINE-TUNING
# ─────────────────────────────────────────────────────────────────────────────

def finetune(df: pd.DataFrame,
             output_dir: str = './finetuned_sbert',
             epochs: int = 3,
             batch_size: int = 16) -> SentenceTransformer:

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.training_args import BatchSamplers
    from datasets import Dataset

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    # New API uses HuggingFace Dataset format
    train_dataset = Dataset.from_dict({
        'anchor':   train_df['query_text'].tolist(),
        'positive': train_df['answer_text'].tolist(),
    })

    model   = SentenceTransformer(BASE_MODEL)
    loss_fn = losses.MultipleNegativesRankingLoss(model)

    warmup_ratio = 0.1
    args = SentenceTransformerTrainingArguments(
        output_dir          = output_dir,
        num_train_epochs    = epochs,
        per_device_train_batch_size = batch_size,
        warmup_ratio        = warmup_ratio,
        batch_sampler       = BatchSamplers.NO_DUPLICATES,
        save_strategy       = 'no',
        logging_strategy    = 'no',   # disables logging that triggers _nested_gather bug
        fp16                = True,
    )

    # Patch for transformers>=4.46 / sentence-transformers mismatch:
    # newer transformers passes num_items_in_batch to compute_loss()
    # which older SentenceTransformerTrainer doesn't accept.
    class _PatchedTrainer(SentenceTransformerTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

    trainer = _PatchedTrainer(
        model         = model,
        args          = args,
        train_dataset = train_dataset,
        loss          = loss_fn,
    )

    print(f"[Finetune] {len(train_df)} train pairs | epochs={epochs} | batch={batch_size}")
    t0 = time.time()
    trainer.train()
    model.save_pretrained(output_dir)
    print(f"[Finetune] Done in {time.time()-t0:.0f}s -> saved to {output_dir}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(data_path: str,
        n: int         = None,
        epochs: int    = 3,
        batch_size: int= 16,
        output_dir: str= './finetuned_sbert',
        skip_finetune: bool = False) -> pd.DataFrame:

    df         = load_tickets(data_path, n=n)
    q_texts    = df['query_text'].tolist()
    a_texts    = df['answer_text'].tolist()
    labels     = df['tag_1'].fillna('unknown').tolist() if 'tag_1' in df.columns else ['?'] * len(df)
    results    = {}

    # ── Pretrained ────────────────────────────────────────────────────────────
    print(f"\n[1/2] Encoding with pretrained {BASE_MODEL}...")
    pre_model  = SentenceTransformer(BASE_MODEL)
    q_pre      = pre_model.encode(q_texts, batch_size=64,
                                  show_progress_bar=True, convert_to_numpy=True)
    a_pre      = pre_model.encode(a_texts, batch_size=64,
                                  show_progress_bar=True, convert_to_numpy=True)

    m_pre      = retrieval_metrics(q_pre, a_pre)
    m_pre['intra_cluster_sim'] = cluster_sim(q_pre, labels)
    results['pretrained'] = m_pre
    print(f"[Pretrained] MRR@10={m_pre['MRR@10']} Hit@1={m_pre['Hit@1']} "
          f"NDCG@10={m_pre['NDCG@10']} cluster={m_pre['intra_cluster_sim']}")

    # ── Fine-tuned ────────────────────────────────────────────────────────────
    if not skip_finetune:
        print(f"\n[2/2] Fine-tuning on (query, answer) pairs...")
        ft_model   = finetune(df, output_dir, epochs, batch_size)
        q_ft       = ft_model.encode(q_texts, batch_size=64,
                                     show_progress_bar=True, convert_to_numpy=True)
        a_ft       = ft_model.encode(a_texts, batch_size=64,
                                     show_progress_bar=True, convert_to_numpy=True)

        m_ft       = retrieval_metrics(q_ft, a_ft)
        m_ft['intra_cluster_sim'] = cluster_sim(q_ft, labels)
        results['finetuned'] = m_ft
        print(f"[Finetuned]  MRR@10={m_ft['MRR@10']} Hit@1={m_ft['Hit@1']} "
              f"NDCG@10={m_ft['NDCG@10']} cluster={m_ft['intra_cluster_sim']}")

        # Delta
        print(f"\n{'='*50}")
        print("  DELTA (finetuned - pretrained):")
        for metric in ['MRR@10', 'Hit@1', 'NDCG@10', 'intra_cluster_sim']:
            d = m_ft[metric] - m_pre[metric]
            print(f"  {metric:22s}: {'+' if d>=0 else ''}{d:.4f}")
        print(f"{'='*50}")

    # ── Save results ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results).T.reset_index(names='model')
    print(f"\n{results_df.to_string(index=False)}")

    out = Path(output_dir) / 'sbert_results.csv'
    out.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out, index=False)
    print(f"\n[Saved] {out}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data',          required=True)
    p.add_argument('--n',             type=int,  default=None,
                   help='Tickets to use (default: all)')
    p.add_argument('--epochs',        type=int,  default=3)
    p.add_argument('--batch_size',    type=int,  default=16)
    p.add_argument('--output_dir',    default='./finetuned_sbert')
    p.add_argument('--skip_finetune', action='store_true',
                   help='Only run pretrained evaluation, skip fine-tuning')
    args = p.parse_args()

    run(
        data_path    = args.data,
        n            = args.n,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        output_dir   = args.output_dir,
        skip_finetune= args.skip_finetune,
    )
