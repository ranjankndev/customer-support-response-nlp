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
from dataclasses import dataclass
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
    model_name     : str
    device         : str
    batch_size     : int
    max_length     : int
    body_max_chars : int


def load_config(path: str) -> IndexerConfig:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
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
        # Recommended: E5 multilingual dense embedder
        model_name     = cfg.get("model_name",     "intfloat/multilingual-e5-large"),
        device         = cfg.get("device",         "cuda"),
        batch_size     = cfg.get("batch_size",     64),
        max_length     = cfg.get("max_length",     512),
        body_max_chars = cfg.get("body_max_chars", 1800),
    )


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER (SentenceTransformers)
# ─────────────────────────────────────────────────────────────────────────────

class STEmbedder:
    def __init__(self, config: IndexerConfig):
        self.config = config
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(config.model_name, device=config.device)

    def embed(self, texts: List[str]) -> np.ndarray:
        # E5: prefixing "passage:" can improve retrieval consistency.
        # Keep it cheap and deterministic (no gradients).
        passages = [f"passage: {t}" for t in texts]
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
        print(f"[Indexer] Loading aspects : {self.config.aspects_csv}")
        aspects = pd.read_csv(self.config.aspects_csv, dtype=str).fillna("")

        print(f"[Indexer] Loading train   : {self.config.train_csv}")
        train   = pd.read_csv(self.config.train_csv,   dtype=str).fillna("")

        # strip whitespace on merge keys to avoid silent mismatches
        merge_keys = [self.config.col_subject,
                      self.config.col_body,
                      self.config.col_language]
        for col in merge_keys:
            aspects[col] = aspects[col].str.strip()
            if col in train.columns:
                train[col]   = train[col].str.strip()

        # bring answer column from train into aspects
        df = aspects.merge(
            train[[self.config.col_subject,
                   self.config.col_body,
                   self.config.col_language,
                   self.config.col_response]],
            on  = merge_keys,
            how = "left",
        )

        total   = len(df)
        covered = df[self.config.col_response].ne("").sum()
        print(f"[Indexer] {total} records | answer coverage: {covered}/{total}")
        if covered < total:
            print(f"[Indexer] WARNING: {total - covered} rows missing answer "
                  f"— check for subject/body/language mismatches between files")

        records = []
        for i, row in df.iterrows():
            ticket_id = (str(row[self.config.col_ticket_id])
                         if self.config.col_ticket_id in df.columns else str(i))

            # store aspects as structured dict — enables clean filtering and reranking
            # e.g. filter intent=="login_or_access" or compare prob_sub directly
            aspects = {
                col: str(row[col]).strip()
                for col in self.config.col_aspects
                if col in df.columns
                and str(row[col]).strip().lower() not in ("", "nan", "none")
            }

            records.append({
                "ticket_id": ticket_id,
                "subject"  : str(row.get(self.config.col_subject,  "")),
                "body"     : str(row.get(self.config.col_body,     "")),
                "response" : str(row.get(self.config.col_response, "")),
                "language" : str(row.get(self.config.col_language, "en")),
                "aspects"  : aspects,   # dict: {prob_sub, prob_statement, cause, intent, ticket_type_nli}
            })
        return records

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
        return {
            "ticket_id": record["ticket_id"],
            "subject"  : record["subject"],
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
        print(f"[Indexer] Done — {index.ntotal} vectors indexed")
        return index

    def run(self):
        records = self.load_records()
        self.build_index(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build FAISS index from aspects CSV")
    parser.add_argument("--config", type=str,
                        default="retrieval_config.yaml")
    args   = parser.parse_args()
    config = load_config(args.config)
    CorpusIndexer(config).run()


if __name__ == "__main__":
    main()
