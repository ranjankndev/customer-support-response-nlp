"""
indexer.py — Build FAISS Index from Aspects CSV
=================================================
INPUT  : aspects_train.csv  (output of aspect_extractor.py)
OUTPUT : faiss_index.bin + metadata.json  (used by retriever.py)

EMBEDDING TEXT : subject + "\\n\\n" + body[:body_max_chars]
  - Language NOT included — use it as a filter at retrieval time if desired
  - Aspects NOT included in embedding — store as metadata for filtering/rerank

METADATA STORED per record:
  - ticket_id  : row identifier
  - subject    : used in aspect_query_vector() and display
  - response   : gold answer fed to generator as RAG context  ← critical
  - language   : used for optional post-retrieval language filter
  - aspects    : used for aspect-based reranking after FAISS retrieval

USAGE:
    python indexer.py --config retrieval_config.yaml
"""

import json
import yaml
import argparse
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IndexerConfig:
    aspects_csv    : str
    train_csv      : str        # original train CSV — merged to bring answer column back
    col_ticket_id  : str
    col_subject    : str
    col_body       : str
    col_response   : str
    col_language   : str
    col_aspects    : List[str]
    index_path     : str
    metadata_path  : str
    # dense matrix saved alongside the index; aspect heads train on it (plan A.1)
    dense_path     : str
    model_name     : str
    device         : str
    batch_size     : int
    max_length     : int
    body_max_chars : int
    # ── plan §6.2 ─────────────────────────────────────────────────────────────
    query_prefix   : str = ""          # E5 wants "query: ", BGE-M3 wants nothing
    passage_prefix : str = ""          # E5 wants "passage: "
    retrieval_mode : str = "dense"     # dense | hybrid (adds BM25)
    bm25_path      : str = "data/index/bm25"
    # gold columns of a past resolved ticket, stored for the aspect bonus (A.3)
    col_gold       : List[str] = field(default_factory=lambda: ["queue", "type",
                                                               "priority"])
    # body kept in metadata for cross-encoder reranking on ticket text (§6.3)
    metadata_body_max_chars: int = 1000
    # rows that must not be in the index (the evaluation samples, §6.2)
    exclude_idx_csvs: List[str] = field(default_factory=list)


def _load_yaml(path: str, overrides_path: str | None = None) -> dict:
    """utils.load_yaml_with_overrides, with a local fallback so indexer.py keeps
    working when it is run as a plain script (python indexer.py)."""
    try:
        from src.utils import load_yaml_with_overrides
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}
        if not (overrides_path and str(overrides_path).strip()):
            return base
        with open(str(overrides_path).strip(), "r", encoding="utf-8") as f:
            overrides = (yaml.safe_load(f) or {}).get("retrieval") or {}
        base.update(overrides)
        return base
    return load_yaml_with_overrides(path, overrides_path, section="retrieval")


def load_config(path: str, overrides_path: str | None = None) -> IndexerConfig:
    """Load retrieval_config.yaml, optionally merging the `retrieval` section of
    an overrides file (config/kaggle.yaml) on top."""
    cfg = _load_yaml(path, overrides_path)
    return IndexerConfig(
        aspects_csv    = cfg["aspects_csv"],
        train_csv      = cfg["train_csv"],
        col_ticket_id  = cfg.get("col_ticket_id",  "ticket_id"),
        col_subject    = cfg.get("col_subject",    "subject"),
        col_body       = cfg.get("col_body",       "body"),
        col_response   = cfg.get("col_response",   "answer"),
        col_language   = cfg.get("col_language",   "language"),
        col_aspects    = cfg.get("col_aspects",    ["prob_sub","prob_statement",
                                                    "cause","intent","ticket_type_nli"]),
        index_path     = cfg["index_path"],
        metadata_path  = cfg["metadata_path"],
        dense_path     = cfg.get("dense_path",     "data/index/dense.npy"),
        # Recommended: E5 multilingual dense embedder
        model_name     = cfg.get("model_name",     "intfloat/multilingual-e5-large"),
        device         = cfg.get("device",         "cuda"),
        batch_size     = cfg.get("batch_size",     64),
        max_length     = cfg.get("max_length",     512),
        body_max_chars = cfg.get("body_max_chars", 1800),
        query_prefix   = cfg.get("query_prefix",   ""),
        passage_prefix = cfg.get("passage_prefix", ""),
        retrieval_mode = cfg.get("retrieval_mode", "dense"),
        bm25_path      = cfg.get("bm25_path",      "data/index/bm25"),
        col_gold       = cfg.get("col_gold",       ["queue", "type", "priority"]),
        metadata_body_max_chars = cfg.get("metadata_body_max_chars", 1000),
        exclude_idx_csvs = cfg.get("exclude_idx_csvs", []) or [],
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER (SentenceTransformers)
# ─────────────────────────────────────────────────────────────────────────────

class STEmbedder:
    def __init__(self, config: IndexerConfig):
        self.config = config
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.model_name, device=config.device)

        max_length = int(getattr(config, "max_length", 0) or 0)
        if max_length:
            self.model.max_seq_length = max_length

    def embed(self, texts: List[str]) -> np.ndarray:
        # The prefix comes from the config (plan §6.2): E5 wants "passage: ",
        # BGE-M3 wants nothing. It used to be hard-coded to the E5 form.
        prefix = self.config.passage_prefix or ""
        passages = [f"{prefix}{t}" for t in texts]
        vecs = self.model.encode(
            passages,
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS INDEXER
# ─────────────────────────────────────────────────────────────────────────────

class CorpusIndexer:
    def __init__(self, config: IndexerConfig):
        self.config   = config
        self.embedder = STEmbedder(config)

    def load_records(self) -> List[dict]:
        print(f"[Indexer] Loading train   : {self.config.train_csv}")
        train = pd.read_csv(self.config.train_csv, dtype=str,
                            low_memory=False).fillna("")

        aspects_csv = str(self.config.aspects_csv or "").strip()
        if not (aspects_csv and Path(aspects_csv).is_file()):
            # Aspects are optional (plan §6.2): without them the index is built
            # from the train CSV directly and every aspect bonus is simply 0.
            print(f"[Indexer] WARNING: no aspects CSV at {aspects_csv or '(unset)'} "
                  f"— indexing the train CSV with empty aspects; the aspect "
                  f"rerank bonus will be 0 for every hit")
            return self._records_from(train, with_aspects=False)

        print(f"[Indexer] Loading aspects : {aspects_csv}")
        aspects = pd.read_csv(aspects_csv, dtype=str).fillna("")

        # strip whitespace on merge keys to avoid silent mismatches
        merge_keys = [self.config.col_subject,
                      self.config.col_body,
                      self.config.col_language]
        for col in merge_keys:
            aspects[col] = aspects[col].str.strip()
            if col in train.columns:
                train[col]   = train[col].str.strip()

        # bring the answer column — and the gold labels — from train into aspects
        carry = [col for col in
                 ([self.config.col_response] + list(self.config.col_gold))
                 if col in train.columns and col not in merge_keys]
        df = aspects.merge(
            train[merge_keys + carry],
            on  = merge_keys,
            how = "left",
        ).fillna("")

        total   = len(df)
        covered = df[self.config.col_response].ne("").sum()
        print(f"[Indexer] {total} records | answer coverage: {covered}/{total}")
        if covered < total:
            print(f"[Indexer] WARNING: {total - covered} rows missing answer "
                  f"— check for subject/body/language mismatches between files")

        return self._records_from(df, with_aspects=True)

    def _records_from(self, df: pd.DataFrame, with_aspects: bool) -> List[dict]:
        """One metadata record per row, with the aspect columns if present."""
        records = []
        for i, row in df.iterrows():
            ticket_id = (str(row[self.config.col_ticket_id])
                         if self.config.col_ticket_id in df.columns
                         else str(row.get("idx", i)))

            # store aspects as structured dict — enables clean filtering and reranking
            # e.g. filter intent=="login_or_access" or compare prob_sub directly
            aspects = {}
            if with_aspects:
                aspects = {
                    col: str(row[col]).strip()
                    for col in self.config.col_aspects
                    if col in df.columns
                    and str(row[col]).strip().lower() not in ("", "nan", "none")
                }
            # A past resolved ticket legitimately carries its gold queue/type/
            # priority, and the query side uses the predicted heads (plan A.3).
            for col in self.config.col_gold:
                value = str(row.get(col, "") or "").strip()
                if value and value.lower() not in ("nan", "none"):
                    aspects[col] = value

            records.append({
                "ticket_id": ticket_id,
                "subject"  : str(row.get(self.config.col_subject,  "")),
                "body"     : str(row.get(self.config.col_body,     "")),
                "response" : str(row.get(self.config.col_response, "")),
                "language" : str(row.get(self.config.col_language, "en")),
                "aspects"  : aspects,
            })
        return records

    def assert_no_evaluation_rows(self, records: List[dict]) -> None:
        """
        The index is built from train only; this makes that an assertion rather
        than an assumption (plan §6.2). Any ticket_id that also appears in an
        evaluation sample would let a system retrieve the row it is scored on.
        """
        if not self.config.exclude_idx_csvs:
            return
        indexed = {str(r["ticket_id"]) for r in records}
        for csv_path in self.config.exclude_idx_csvs:
            path = Path(csv_path)
            if not path.is_file():
                print(f"[Indexer] WARNING: exclude list not found: {path}")
                continue
            sample = pd.read_csv(path, dtype=str).fillna("")
            column = ("ticket_id" if "ticket_id" in sample.columns
                      else "idx" if "idx" in sample.columns else None)
            if column is None:
                print(f"[Indexer] WARNING: {path} has no ticket_id/idx column")
                continue
            overlap = indexed & set(sample[column].astype(str))
            if overlap:
                raise AssertionError(
                    f"{len(overlap)} evaluation rows from {path} are in the "
                    f"index (e.g. {sorted(overlap)[:5]}): the index must be "
                    f"built from the train split only."
                )
            print(f"[Indexer] No overlap with {path} ({len(sample)} rows)")

    def _embedding_text(self, record: dict) -> str:
        """
        Text fed to the embedder.
        Only subject + body — language excluded (use for optional filter).
        Aspects excluded from embedding text — store for filtering/rerank only.
        """
        body_snippet = record["body"][:self.config.body_max_chars] if self.config.body_max_chars else record["body"]
        return f"{record['subject']}\n\n{body_snippet}"

    def _metadata_payload(self, record: dict) -> dict:
        """
        What gets stored in metadata.json per record.
        subject  — needed for aspect_query_vector() in retriever
        response — fed to generator as RAG context
        language — used for optional post-retrieval language filter
        aspects  — used for aspect-based reranking after FAISS retrieval
        body     — NOT stored (too large, not needed at retrieval time)
        """
        # body IS stored now, capped at metadata_body_max_chars: the multilingual
        # reranker scores ticket text against ticket text (plan §6.2/6.3), and a
        # 1000-character cap keeps metadata.json small.
        cap = int(self.config.metadata_body_max_chars or 0)
        body = record["body"][:cap] if cap else record["body"]
        return {
            "ticket_id": record["ticket_id"],
            "subject"  : record["subject"],
            "body"     : body,
            "response" : record["response"],
            "language" : record["language"],
            "aspects"  : record["aspects"],
        }

    def build_index(self, records: List[dict]) -> faiss.Index:
        texts = [self._embedding_text(r) for r in records]

        print(f"[Indexer] Embedding {len(texts)} records with {self.config.model_name} ...")
        dense_vecs = self.embedder.embed(texts)   # [N, d] already normalized

        index = faiss.IndexFlatIP(dense_vecs.shape[1])
        index.add(dense_vecs)

        # save FAISS index
        Path(self.config.index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, self.config.index_path)
        print(f"[Indexer] FAISS index saved  → {self.config.index_path}")

        # save metadata
        metadata = [self._metadata_payload(r) for r in records]
        Path(self.config.metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"[Indexer] Metadata saved     → {self.config.metadata_path}")

        # save the dense matrix (plan A.1): the aspect heads train on exactly these
        # embeddings, so they must not be recomputed with a different model.
        # Row order == metadata order.
        if self.config.dense_path:
            dense_path = Path(self.config.dense_path)
            dense_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(dense_path, dense_vecs)
            print(f"[Indexer] Dense matrix saved → {dense_path} {dense_vecs.shape}")

        # ── BM25, same row order as FAISS (plan §6.2) ─────────────────────────
        if str(self.config.retrieval_mode or "dense").lower() == "hybrid":
            self.build_bm25(texts)

        print(f"[Indexer] Done — {index.ntotal} vectors indexed")
        return index

    def build_bm25(self, texts: List[str]) -> None:
        """
        Lexical half of hybrid retrieval. Dense embeddings generalise but miss
        literal strings (error codes, product names, invoice numbers), which is
        exactly what tickets quote. Row order matches FAISS, so a BM25 document
        id indexes the same metadata entry.
        """
        import bm25s

        path = Path(self.config.bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[Indexer] Building BM25 over {len(texts)} documents ...")
        tokens = bm25s.tokenize(texts, stopwords=None)
        retriever = bm25s.BM25()
        retriever.index(tokens)
        retriever.save(str(path))
        print(f"[Indexer] BM25 saved         → {path}")

    def run(self):
        records = self.load_records()
        self.assert_no_evaluation_rows(records)
        self.build_index(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from aspects CSV")
    parser.add_argument("--config", type=str,
                        default="retrieval_config.yaml")
    parser.add_argument("--overrides", type=str, default=None,
                        help="YAML whose `retrieval` section is merged on top of "
                             "--config (e.g. config/kaggle.yaml)")
    args   = parser.parse_args()
    config = load_config(args.config, args.overrides)
    CorpusIndexer(config).run()


if __name__ == "__main__":
    main()
