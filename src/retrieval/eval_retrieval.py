"""
eval_retrieval.py — retrieval-only proxy metrics (plan §6.4)
============================================================
Generation metrics cannot tell a retrieval regression from a generation one, so
the retrieval stack is scored on its own, on val_500, before any GPU time is
spent on replies.

Per row (top-3 hits):
  top1_rougeL  Unicode ROUGE-L between the top-1 retrieved response and the
               gold answer -- "did we retrieve an answer that looks like the
               right one"
  top1_cos     cosine between the same pair, with the index's own embedder --
               the same question without lexical overlap
  intent@3     share of the top-3 hits whose stored intent (or gold queue)
               equals the query's intent; skipped when either side has none

USAGE
    python -m src.retrieval.eval_retrieval \
        --config config/inference_retrieval_config.yaml \
        --input data/minimal/val_500_with_aspects.csv --label m3_hybrid_rerank
    # compare configurations by passing a different --config/--overrides and
    # the same --out; rows accumulate per label.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

OUT_DEFAULT = "output/report/retrieval_table.csv"
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
METRICS = ("top1_rougeL", "top1_cos", "intent@3")


# ─────────────────────────────────────────────────────────────────────────────
# PER-ROW METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RougeCfg:
    rouge_types: List[str] = field(default_factory=lambda: ["rougeL"])
    rouge_unicode_tokenizer: bool = True
    rouge_use_stemmer: bool = False


def rouge_l(hypothesis: str, reference: str) -> float:
    """Unicode ROUGE-L for one pair (the same scorer the report uses)."""
    from src.utils import MetricsCalculator

    scores = MetricsCalculator(_RougeCfg()).rouge([hypothesis or ""],
                                                  [reference or ""])
    return float(scores["rougeL"])


def cosine(embedder, text_a: str, text_b: str) -> float:
    """Cosine between two texts, embedded with the index's own model."""
    vectors = embedder.encode([text_a or "", text_b or ""], kind="passage")
    a, b = np.asarray(vectors[0], dtype=np.float32), np.asarray(vectors[1],
                                                                dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _hit_intent(hit: Dict) -> str:
    aspects = hit.get("aspects") or {}
    for key in ("intent", "queue"):
        value = str(aspects.get(key, "") or "").strip().lower()
        if value and value != "none":
            return value
    return ""


def intent_at_k(query_intent: str, hits: List[Dict]) -> Optional[float]:
    """
    Share of hits whose intent matches the query's. None (not 0) when the query
    has no intent or the index carries none, so rows that cannot be judged are
    excluded from the mean instead of dragging it down.
    """
    wanted = str(query_intent or "").strip().lower()
    if not wanted or wanted == "none" or not hits:
        return None
    hit_intents = [_hit_intent(hit) for hit in hits]
    if not any(hit_intents):
        return None
    return sum(1 for intent in hit_intents if intent == wanted) / len(hits)


def evaluate_rows(retriever, df: pd.DataFrame, top_k: int = 3,
                  exclude_self: bool = True) -> pd.DataFrame:
    """One row of metrics per query row."""
    from src.minimal_run import ASPECT_COLUMNS

    records = []
    for position, row in df.iterrows():
        subject = str(row.get("subject", "") or "")
        body = str(row.get("body", "") or "")
        language = str(row.get("language", "en") or "en")
        gold = str(row.get("answer", "") or "")
        spans = {column: str(row.get(column, "") or "")
                 for column in ASPECT_COLUMNS if column in df.columns}
        spans = {k: v for k, v in spans.items()
                 if v and v.lower() not in ("none", "nan")}

        # a val row must never retrieve itself, even if the index was rebuilt
        exclude = ({str(row.get("idx", position))} if exclude_self else None)
        hits = retriever.retrieve(subject=subject, body=body, language=language,
                                  aspect_spans=(spans or None), top_k=top_k,
                                  exclude_ids=exclude)

        top1 = hits[0] if hits else {}
        top1_response = str(top1.get("response", "") or "")
        records.append({
            "idx"        : row.get("idx", position),
            "language"   : language.lower()[:2],
            "n_hits"     : len(hits),
            "top1_rougeL": rouge_l(top1_response, gold) if hits else 0.0,
            "top1_cos"   : (cosine(retriever.embedder, top1_response, gold)
                            if hits and gold else 0.0),
            "intent@3"   : intent_at_k(spans.get("intent", ""), hits),
            "top1_ce"    : top1.get("ce_score", ""),
        })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_mean(values: List[float], resamples: int = BOOTSTRAP_RESAMPLES,
                   seed: int = BOOTSTRAP_SEED) -> Dict[str, float]:
    """Mean with a 95% percentile bootstrap CI (seeded, so it is reproducible)."""
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return {"mean": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": 0}
    array = np.asarray(clean, dtype=np.float64)
    if len(array) == 1:
        return {"mean": float(array[0]), "ci_low": float(array[0]),
                "ci_high": float(array[0]), "n": 1}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    means = array[draws].mean(axis=1)
    return {"mean": round(float(array.mean()), 4),
            "ci_low": round(float(np.percentile(means, 2.5)), 4),
            "ci_high": round(float(np.percentile(means, 97.5)), 4),
            "n": len(array)}


def summarise(per_row: pd.DataFrame, label: str) -> pd.DataFrame:
    """Mean + CI per metric, overall and per language."""
    groups = [("all", per_row)]
    for language in sorted(set(per_row.get("language", pd.Series(dtype=str)))):
        if language:
            groups.append((language, per_row[per_row["language"] == language]))

    rows = []
    for split, frame in groups:
        for metric in METRICS:
            if metric not in frame.columns:
                continue
            stats = bootstrap_mean(frame[metric].tolist())
            rows.append({"config": label, "split": split, "metric": metric,
                         **stats})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def run(config_path: str, input_path: str, label: str,
        out_path: str = OUT_DEFAULT, top_k: int = 3,
        n: Optional[int] = None, overrides_path: Optional[str] = None,
        per_row_path: Optional[str] = None) -> pd.DataFrame:
    from src.retrieval.inference_retriever import load_retriever_config
    from src.retrieval.retriever import TicketRetriever

    df = pd.read_csv(input_path, dtype=str, low_memory=False).fillna("")
    if n:
        df = df.head(n)
    print(f"[EvalRetrieval] {label}: {len(df)} rows from {input_path}")

    retriever = TicketRetriever(load_retriever_config(config_path,
                                                      overrides_path))
    per_row = evaluate_rows(retriever, df, top_k=top_k)
    table = summarise(per_row, label)
    print(table.to_string(index=False))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file():
        # keep the other configurations' rows, replace this label's
        previous = pd.read_csv(out)
        previous = previous[previous.get("config", "") != label]
        table = pd.concat([previous, table], ignore_index=True)
    table.to_csv(out, index=False)
    print(f"[EvalRetrieval] Retrieval table -> {out}")

    if per_row_path:
        per_row_out = Path(per_row_path)
        per_row_out.parent.mkdir(parents=True, exist_ok=True)
        per_row.to_csv(per_row_out, index=False)
        print(f"[EvalRetrieval] Per-row metrics -> {per_row_out}")
    return table


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/inference_retrieval_config.yaml")
    parser.add_argument("--overrides", default=None,
                        help="YAML whose `retrieval` section is merged on top")
    parser.add_argument("--input", required=True,
                        help="val_500 with aspect columns")
    parser.add_argument("--label", required=True,
                        help="e5_dense | m3_dense | m3_hybrid | m3_hybrid_rerank")
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--per_row_out", default=None)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--n", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        run(args.config, args.input, args.label, args.out, args.top_k, args.n,
            args.overrides, args.per_row_out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
