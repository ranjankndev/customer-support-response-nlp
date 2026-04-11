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

from dataclasses import dataclass, fields
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

    model_name: str = "intfloat/multilingual-e5-large"
    device: str = "cuda"
    batch_size: int = 64
    max_length: int = 512  # kept for compatibility; ST embedder ignores this unless you wire it in
    top_k: int = 5
    body_max_chars: int = 0           # 0 = full body for retrieval embedding
    # prompt-side truncation (independent of retrieval embedding truncation)
    # 0 = full body in prompts
    prompt_body_max_chars: int = 0
    filter_by_language: bool = False
    aspect_rerank_weight: float = 0.1    # reduced from 0.25 -- prevents aspect bonus flipping ranking
    use_cross_encoder: bool = True        # EN: cross-encoder reranking; DE: pure dense
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def load_retriever_config(path: str) -> RetrievalConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    allowed = {f.name for f in fields(RetrievalConfig)}
    filtered = {k: v for k, v in cfg.items() if k in allowed}
    return RetrievalConfig(**filtered)


_retriever: Optional[TicketRetriever] = None
_retriever_cfg_path: Optional[str] = None


def get_retriever(config_path: str = "inference_retrieval_config.yaml") -> TicketRetriever:
    """
    Cached retriever instance (loads FAISS + metadata + embedding model once).
    """
    global _retriever, _retriever_cfg_path
    if _retriever is None or _retriever_cfg_path != config_path:
        cfg = load_retriever_config(config_path)
        _retriever = TicketRetriever(cfg)
        _retriever_cfg_path = config_path
    return _retriever


def retrieve_similar(
    subject: str,
    body: str,
    language: str = "en",
    config_path: str = "inference_retrieval_config.yaml",
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieval-only helper (NO aspects required).
    Returns a list of dicts: subject/response/language/aspects + dense/final scores.
    """
    retriever = get_retriever(config_path)
    return retriever.retrieve(
        subject=subject,
        body=body,
        language=language,
        aspect_spans=None,
        top_k=top_k,
    )


def run_file(
    config_path: str = "inference_retrieval_config.yaml",
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    n: Optional[int] = None,
) -> str:
    """
    Batch retrieval for a CSV/XLSX file.
    Reads subject/body/(language) and writes top-k hits per row.
    """
    cfg = load_retriever_config(config_path)
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

    retriever = get_retriever(config_path)
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
                "final": h.get("final"),
            })

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main():
    import argparse

    p = argparse.ArgumentParser(description="FAISS retrieval for inference tickets (no aspects required)")
    p.add_argument("--config", default="inference_retrieval_config.yaml")
    p.add_argument("--input", default=None, help="Override config input_path")
    p.add_argument("--out", default=None, help="Override config output_path")
    p.add_argument("--n", type=int, default=None, help="Sample N rows")
    args = p.parse_args()

    out = run_file(config_path=args.config, input_path=args.input, output_path=args.out, n=args.n)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

