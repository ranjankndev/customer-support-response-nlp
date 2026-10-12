# -*- coding: utf-8 -*-
"""
inference_retriever.py

Tiny helper to load `inference_retrieval_config.yaml` and run FAISS retrieval.
Aspects are OPTIONAL and NOT required for retrieval.

Example:
  from inference_retriever import retrieve_similar
  hits = retrieve_similar(
      subject="Login not working",
      body="I cannot access my account since yesterday...",
      language="en",
      config_path="inference_retrieval_config.yaml",
      top_k=3,
  )
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, List, Dict, Any

import yaml
import pandas as pd
from pathlib import Path

from src.retrieval.retriever import TicketRetriever


@dataclass
class RetrievalConfig:
    # Required first (dataclass rule: no-default fields before defaulted fields)
    index_path: str
    metadata_path: str
    # I/O (optional in YAML for library-style use)
    input_path: Optional[str] = None
    output_path: Optional[str] = None
    col_subject: str = "subject"
    col_body: str = "body"
    col_language: str = "language"

    model_name: str = "BAAI/bge-m3"
    # prefixes must match the indexer's (plan §6.2/6.3)
    query_prefix: str = ""
    passage_prefix: str = ""
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 512          # sets SentenceTransformer.max_seq_length
    top_k: int = 5
    body_max_chars: int = 0           # 0 = full body for retrieval embedding
    # prompt-side truncation (independent of retrieval embedding truncation)
    # 0 = full body in prompts
    prompt_body_max_chars: int = 0
    filter_by_language: bool = False
    aspect_rerank_weight: float = 0.1    # reduced from 0.25 -- prevents aspect bonus flipping ranking
    # ── hybrid retrieval (plan §6.2/6.3) ──────────────────────────────────────
    retrieval_mode: str = "dense"        # dense | hybrid (adds BM25 + RRF)
    bm25_path: str = "data/index/bm25"
    rrf_k: int = 60
    n_candidates: int = 50
    # ── reranking ─────────────────────────────────────────────────────────────
    use_cross_encoder: bool = True
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    cross_encoder_max_length: int = 512
    rerank_languages: List[str] = field(default_factory=lambda: ["en", "de"])
    rerank_doc_field: str = "body"       # body | response
    rerank_batch_size: int = 64
    # languages whose aspect spans are trusted enough to weight the query vector
    aspect_query_languages: List[str] = field(default_factory=lambda: ["en"])
    # checklist = aspect instructions (plan A.3) | slots = legacy "Aspects : k: v" dump
    aspect_prompt_style: str = "checklist"
    # prompt budget: max chars per retrieved response (0 = full); plan §5.3
    ctx_doc_max_chars: int = 0
    # append "Write the reply in German, ..." before "Support reply:"; plan §5.4
    reply_language_instruction: bool = True


def load_retriever_config(path: str,
                          overrides_path: Optional[str] = None) -> RetrievalConfig:
    """Load the retrieval config, optionally merging the `retrieval` section of
    an overrides file (config/kaggle.yaml) on top."""
    try:
        from src.utils import load_yaml_with_overrides

        cfg = load_yaml_with_overrides(path, overrides_path, section="retrieval")
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    allowed = {f.name for f in fields(RetrievalConfig)}
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return RetrievalConfig(**filtered)


_retriever: Optional[TicketRetriever] = None
_retriever_cfg_path: Optional[str] = None


def get_retriever(config_path: str = "inference_retrieval_config.yaml",
                  overrides_path: Optional[str] = None) -> TicketRetriever:
    """
    Cached retriever instance (loads FAISS + metadata + embedding model once).
    """
    global _retriever, _retriever_cfg_path
    key = (config_path, overrides_path)
    if _retriever is None or _retriever_cfg_path != key:
        cfg = load_retriever_config(config_path, overrides_path)
        _retriever = TicketRetriever(cfg)
        _retriever_cfg_path = key
    return _retriever


def retrieve_similar(
    subject: str,
    body: str,
    language: str = "en",
    config_path: str = "inference_retrieval_config.yaml",
    top_k: Optional[int] = None,
    exclude_ids: Optional[set] = None,
    overrides_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval-only helper (NO aspects required).
    Returns a list of dicts: subject/response/language/aspects + dense/final scores.
    """
    retriever = get_retriever(config_path, overrides_path)
    return retriever.retrieve(
        subject=subject,
        body=body,
        language=language,
        aspect_spans=None,
        top_k=top_k,
        exclude_ids=exclude_ids,
    )


def run_file(
    config_path: str = "inference_retrieval_config.yaml",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    n: Optional[int] = None,
    overrides_path: Optional[str] = None,
) -> str:
    """
    Batch retrieval for a CSV/XLSX file.
    Reads subject/body/(language) and writes top-k hits per row.
    """
    cfg = load_retriever_config(config_path, overrides_path)
    in_path = input_path or cfg.input_path
    if not in_path:
        raise ValueError("Missing input_path (set in YAML or pass --input).")
    out_path = output_path or cfg.output_path or "output/retrieval_hits.csv"

    df = pd.read_excel(in_path) if str(in_path).lower().endswith((".xlsx", ".xls")) else pd.read_csv(in_path, low_memory=False)
    if n:
        df = df.sample(min(n, len(df)), random_state=42).reset_index(drop=True)

    for c in (cfg.col_subject, cfg.col_body):
        if c not in df.columns:
            raise ValueError(f"Input missing required column: {c}. Columns: {list(df.columns)}")
    if cfg.col_language not in df.columns:
        df[cfg.col_language] = "en"

    retriever = get_retriever(config_path, overrides_path)
    top_k = int(getattr(cfg, "top_k", 5) or 5)

    rows: List[Dict[str, Any]] = []
    for i, row in df.iterrows():
        subj = str(row.get(cfg.col_subject, "") or "")
        body = str(row.get(cfg.col_body, "") or "")
        lang = str(row.get(cfg.col_language, "en") or "en")

        hits = retriever.retrieve(
            subject=subj,
            body=body,
            language=lang,
            aspect_spans=None,
            top_k=top_k,
        )
        for rank, h in enumerate(hits, 1):
            rows.append({
                "row_idx": i,
                "rank": rank,
                "query_language": lang,
                "query_subject": subj,
                "hit_ticket_idx": h.get("idx"),
                "hit_language": h.get("language"),
                "hit_subject": h.get("subject"),
                "hit_response": h.get("response"),
                "dense": h.get("dense"),
                "rrf": h.get("rrf"),
                "ce_score": h.get("ce_score"),
                "a_score": h.get("a_score"),
                "final": h.get("final"),
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="FAISS retrieval for inference tickets (no aspects required)")
    p.add_argument("--config", default="inference_retrieval_config.yaml")
    p.add_argument("--overrides", default=None,
                   help="YAML whose `retrieval` section is merged on top of --config")
    p.add_argument("--input", default=None, help="Override config input_path")
    p.add_argument("--out", default=None, help="Override config output_path")
    p.add_argument("--n", type=int, default=None, help="Sample N rows")
    args = p.parse_args()

    out = run_file(config_path=args.config, input_path=args.input,
                   output_path=args.out, n=args.n, overrides_path=args.overrides)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

