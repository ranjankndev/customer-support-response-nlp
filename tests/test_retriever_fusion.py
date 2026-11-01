"""Step 5 (plan §6.2–6.3): RRF fusion, hybrid retrieval, the multilingual
reranker, exclude_ids and the aspect bonus. A tiny FAISS index + BM25 index are
built in a temp directory with a stub embedder, so no model is downloaded."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import faiss
import numpy as np
import pytest

from src.retrieval.retriever import TicketRetriever, aspect_score, rrf_fuse


# ── RRF on toy ranks ─────────────────────────────────────────────────────────

def test_rrf_scores_one_list_by_reciprocal_rank():
    fused = rrf_fuse([[7, 3, 1]], k=60)

    assert fused[7] == pytest.approx(1 / 61)
    assert fused[3] == pytest.approx(1 / 62)
    assert fused[1] == pytest.approx(1 / 63)


def test_rrf_rewards_a_document_both_lists_agree_on():
    dense = [1, 2, 3]
    sparse = [3, 2, 9]

    fused = rrf_fuse([dense, sparse], k=60)

    # 2 and 3 appear in both lists, 1 and 9 in only one, so the agreed-on
    # documents come out on top
    assert min(fused[2], fused[3]) > max(fused[1], fused[9])
    # being first in one list is worth slightly more than second in both
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62 + 1 / 62)
    assert fused[9] == pytest.approx(1 / 63)


def test_rrf_k_damps_the_head_of_each_list():
    small_k = rrf_fuse([[1], [2]], k=1)
    large_k = rrf_fuse([[1], [2]], k=1000)

    assert small_k[1] == pytest.approx(0.5)
    assert large_k[1] == pytest.approx(1 / 1001)
    # with a large k the gap between rank 1 and rank 2 almost disappears
    ranked = rrf_fuse([[1, 2]], k=1000)
    assert ranked[1] - ranked[2] < 1e-5


def test_rrf_of_nothing_is_empty():
    assert rrf_fuse([], k=60) == {}
    assert rrf_fuse([[]], k=60) == {}


# ── aspect_score ─────────────────────────────────────────────────────────────

def test_aspect_score_weights_intent_type_and_overlap():
    query = {"intent": "Billing", "ticket_type": "Incident",
             "prob_sub": "Die Rechnung ist falsch", "cause": "none"}
    doc = {"queue": "billing", "type": "incident",
           "prob_sub": "Die Rechnung ist falsch"}

    # 0.4 intent + 0.2 type + 0.3 full prob_sub overlap
    assert aspect_score(query, doc) == pytest.approx(0.9)


def test_aspect_score_uses_unicode_tokens_for_overlap():
    """"Größe" and "Grüße" are different words and must not overlap."""
    assert aspect_score({"prob_sub": "Größe"}, {"prob_sub": "Grüße"}) == 0.0
    assert aspect_score({"prob_sub": "Größe ändern"},
                        {"prob_sub": "Größe ändern"}) == pytest.approx(0.3)


def test_aspect_score_matches_predicted_heads_against_gold_metadata():
    """The query carries predicted `intent`; a train document carries gold
    `queue` (plan A.3)."""
    assert aspect_score({"intent": "Technical Support"},
                        {"queue": "technical support"}) == pytest.approx(0.4)


def test_aspect_score_ignores_none_values():
    assert aspect_score({"intent": "none"}, {"queue": "none"}) == 0.0
    assert aspect_score({}, {"queue": "billing"}) == 0.0
    assert aspect_score({"intent": "billing"}, {}) == 0.0


# ── a tiny index to retrieve from ────────────────────────────────────────────

DOCS = [
    {"ticket_id": "t0", "subject": "Login fails", "body": "error ERR-500 on login",
     "response": "Reset your password", "language": "en",
     "aspects": {"queue": "Technical Support", "type": "Incident",
                 "prob_sub": "login fails"}},
    {"ticket_id": "t1", "subject": "Invoice wrong", "body": "charged twice",
     "response": "We refunded the duplicate", "language": "en",
     "aspects": {"queue": "Billing", "type": "Request",
                 "prob_sub": "invoice wrong"}},
    {"ticket_id": "t2", "subject": "Anmeldung schlägt fehl",
     "body": "Fehler ERR-500 bei der Anmeldung",
     "response": "Bitte setzen Sie Ihr Passwort zurück", "language": "de",
     "aspects": {"queue": "Technical Support", "type": "Incident",
                 "prob_sub": "anmeldung schlägt fehl"}},
]

# one-hot vectors: doc i is closest to query vector i
VECTORS = np.eye(len(DOCS), dtype=np.float32)


@dataclass
class StubConfig:
    index_path: str = ""
    metadata_path: str = ""
    bm25_path: str = ""
    top_k: int = 2
    n_candidates: int = 10
    rrf_k: int = 60
    retrieval_mode: str = "dense"
    body_max_chars: int = 0
    filter_by_language: bool = False
    use_cross_encoder: bool = False
    cross_encoder_model: str = "stub"
    cross_encoder_max_length: int = 512
    rerank_languages: List[str] = field(default_factory=lambda: ["en", "de"])
    rerank_doc_field: str = "body"
    rerank_batch_size: int = 64
    aspect_query_languages: List[str] = field(default_factory=lambda: ["en"])
    aspect_rerank_weight: float = 0.1
    model_name: str = "stub"
    device: str = "cpu"
    query_prefix: str = ""
    passage_prefix: str = ""
    max_length: int = 0
    aspect_prompt_style: str = "checklist"
    ctx_doc_max_chars: int = 0
    reply_language_instruction: bool = True


class _StubEmbedder:
    """Maps a query to the one-hot vector of the document it should match."""

    def __init__(self, config=None):
        self.config = config

    def encode(self, texts, kind="passage"):
        out = []
        for text in texts:
            lowered = text.lower()
            if "anmeldung" in lowered or "deutsch" in lowered:
                out.append(VECTORS[2])
            elif "invoice" in lowered or "rechnung" in lowered:
                out.append(VECTORS[1])
            else:
                out.append(VECTORS[0])
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def index_dir(tmp_path):
    index = faiss.IndexFlatIP(len(DOCS))
    index.add(VECTORS)
    index_path = tmp_path / "faiss.bin"
    faiss.write_index(index, str(index_path))
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(DOCS, ensure_ascii=False),
                             encoding="utf-8")

    import bm25s

    texts = [f"{d['subject']}\n\n{d['body']}" for d in DOCS]
    bm25 = bm25s.BM25()
    bm25.index(bm25s.tokenize(texts, stopwords=None))
    bm25_path = tmp_path / "bm25"
    bm25.save(str(bm25_path))

    return {"index": str(index_path), "metadata": str(metadata_path),
            "bm25": str(bm25_path)}


def _retriever(monkeypatch, index_dir, **overrides):
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "STEmbedder", _StubEmbedder)
    overrides.setdefault("bm25_path", index_dir["bm25"])
    config = StubConfig(index_path=index_dir["index"],
                        metadata_path=index_dir["metadata"], **overrides)
    return TicketRetriever(config)


# ── dense and hybrid ─────────────────────────────────────────────────────────

def test_dense_retrieval_returns_the_matching_document(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir)

    hits = retriever.retrieve("Login fails", "error ERR-500", "en")

    assert hits[0]["ticket_id"] == "t0"
    assert hits[0]["dense"] == pytest.approx(1.0)
    assert "rrf" in hits[0]
    assert len(hits) == 2          # top_k


def test_hybrid_mode_loads_bm25_and_fuses(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir, retrieval_mode="hybrid")

    assert retriever.bm25 is not None
    assert retriever.bm25_mode == "hybrid"

    hits = retriever.retrieve("Login fails", "error ERR-500 on login", "en")

    assert hits[0]["ticket_id"] == "t0"
    # BM25 contributed: the exact string "ERR-500" appears in t0 and t2
    assert {h["ticket_id"] for h in hits} <= {"t0", "t1", "t2"}


def test_hybrid_falls_back_to_dense_without_a_bm25_index(monkeypatch, index_dir,
                                                         tmp_path):
    retriever = _retriever(monkeypatch, index_dir, retrieval_mode="hybrid",
                           bm25_path=str(tmp_path / "missing"))

    assert retriever.bm25 is None
    assert retriever.bm25_mode == "dense"
    assert retriever.retrieve("Login fails", "x", "en")[0]["ticket_id"] == "t0"


def test_hybrid_finds_a_literal_string_dense_retrieval_would_miss(monkeypatch,
                                                                  index_dir):
    """The point of the lexical half: the stub embedder sends this query to t1,
    but the quoted error code only appears in t0 and t2."""
    retriever = _retriever(monkeypatch, index_dir, retrieval_mode="hybrid",
                           top_k=3)

    hits = retriever.retrieve("Invoice question", "we also see ERR-500", "en")

    assert {h["ticket_id"] for h in hits} >= {"t0", "t1"}


# ── exclude_ids ──────────────────────────────────────────────────────────────

def test_exclude_ids_removes_the_given_ticket(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir, top_k=3)

    hits = retriever.retrieve("Login fails", "error", "en", exclude_ids={"t0"})

    assert "t0" not in {h["ticket_id"] for h in hits}
    assert len(hits) == 2


def test_exclude_ids_accepts_non_string_ids(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir, top_k=3)
    DOCS_BY_ID = {d["ticket_id"]: d for d in DOCS}

    hits = retriever.retrieve("Login fails", "error", "en", exclude_ids={0, "t1"})

    assert all(h["ticket_id"] in DOCS_BY_ID for h in hits)
    assert "t1" not in {h["ticket_id"] for h in hits}


# ── language filter ──────────────────────────────────────────────────────────

def test_filter_by_language_keeps_only_the_query_language(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir, filter_by_language=True,
                           top_k=3)

    hits = retriever.retrieve("Anmeldung schlägt fehl", "Fehler", "de")

    assert {h["language"] for h in hits} == {"de"}


# ── reranking ────────────────────────────────────────────────────────────────

class _StubCrossEncoder:
    """Scores a pair by how much of the query's text the document repeats."""

    def __init__(self):
        self.seen_pairs = []
        self.batch_sizes = []

    def predict(self, pairs, batch_size=None):
        self.seen_pairs.extend(pairs)
        self.batch_sizes.append(batch_size)
        scores = []
        for query, doc in pairs:
            q_tokens = set(query.lower().split())
            d_tokens = set(doc.lower().split())
            scores.append(len(q_tokens & d_tokens) / max(len(q_tokens), 1))
        return scores


def test_german_rows_are_reranked_too(monkeypatch, index_dir):
    """v2 skipped the cross-encoder for DE, so DE rows had no ce_score at all."""
    import src.retrieval.retriever as retriever_module

    encoder = _StubCrossEncoder()
    monkeypatch.setattr(retriever_module, "get_cross_encoder",
                        lambda *a, **k: encoder)
    retriever = _retriever(monkeypatch, index_dir, use_cross_encoder=True,
                           top_k=3)

    hits = retriever.retrieve("Anmeldung schlägt fehl",
                              "Fehler ERR-500 bei der Anmeldung", "de")

    assert all(isinstance(h["ce_score"], float) for h in hits)
    assert hits[0]["ticket_id"] == "t2"


def test_rerank_languages_can_exclude_a_language(monkeypatch, index_dir):
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_cross_encoder",
                        lambda *a, **k: _StubCrossEncoder())
    retriever = _retriever(monkeypatch, index_dir, use_cross_encoder=True,
                           rerank_languages=["en"], top_k=2)

    german = retriever.retrieve("Anmeldung", "Fehler", "de")
    english = retriever.retrieve("Login fails", "error", "en")

    assert all(h["ce_score"] == "" for h in german)
    assert all(isinstance(h["ce_score"], float) for h in english)


def test_rerank_doc_field_chooses_ticket_or_answer_text(monkeypatch, index_dir):
    import src.retrieval.retriever as retriever_module

    encoder = _StubCrossEncoder()
    monkeypatch.setattr(retriever_module, "get_cross_encoder",
                        lambda *a, **k: encoder)

    retriever = _retriever(monkeypatch, index_dir, use_cross_encoder=True,
                           rerank_doc_field="response", top_k=1)
    retriever.retrieve("Login fails", "error", "en")

    assert any("Reset your password" in doc for _, doc in encoder.seen_pairs)
    assert not any("ERR-500 on login" in doc for _, doc in encoder.seen_pairs)


def test_the_reranker_is_called_with_the_configured_batch_size(monkeypatch,
                                                               index_dir):
    import src.retrieval.retriever as retriever_module

    encoder = _StubCrossEncoder()
    monkeypatch.setattr(retriever_module, "get_cross_encoder",
                        lambda *a, **k: encoder)
    retriever = _retriever(monkeypatch, index_dir, use_cross_encoder=True,
                           rerank_batch_size=8)
    retriever.retrieve("Login fails", "error", "en")

    assert encoder.batch_sizes == [8]


# ── the aspect bonus is applied after reranking, for both languages ──────────

def test_the_aspect_bonus_is_added_on_top_of_the_rerank_score(monkeypatch,
                                                              index_dir):
    """v2 dropped the aspect term entirely once the cross-encoder had run."""
    import src.retrieval.retriever as retriever_module

    monkeypatch.setattr(retriever_module, "get_cross_encoder",
                        lambda *a, **k: _StubCrossEncoder())
    retriever = _retriever(monkeypatch, index_dir, use_cross_encoder=True,
                           aspect_rerank_weight=0.5, top_k=3)

    spans = {"intent": "Technical Support", "ticket_type": "Incident",
             "prob_sub": "login fails"}
    hits = retriever.retrieve("Login fails", "error", "en", aspect_spans=spans)

    scored = {h["ticket_id"]: h for h in hits}
    assert scored["t0"]["a_score"] > 0
    assert scored["t0"]["final"] == pytest.approx(
        scored["t0"]["ce_score"] + 0.5 * scored["t0"]["a_score"])
    assert scored["t1"]["a_score"] == 0.0


def test_without_aspects_the_bonus_is_zero(monkeypatch, index_dir):
    retriever = _retriever(monkeypatch, index_dir, aspect_rerank_weight=0.5)

    hits = retriever.retrieve("Login fails", "error", "en", aspect_spans=None)

    assert all(h["a_score"] == 0.0 for h in hits)
    assert hits[0]["final"] == pytest.approx(hits[0]["dense"])


def test_aspect_query_languages_decides_which_queries_are_aspect_weighted(
        monkeypatch, index_dir):
    """DE aspect spans are noisier, so by default only EN queries are weighted
    by them — but DE still gets the rerank and the bonus."""
    import src.retrieval.retriever as retriever_module

    weighted = []
    monkeypatch.setattr(retriever_module, "aspect_query_vector",
                        lambda embedder, subject, body, spans:
                        (weighted.append(subject) or VECTORS[0]))
    retriever = _retriever(monkeypatch, index_dir,
                           aspect_query_languages=["en"])
    spans = {"prob_sub": "login fails", "intent": "Technical Support"}

    retriever.retrieve("Login fails", "error", "en", aspect_spans=spans)
    retriever.retrieve("Anmeldung", "Fehler", "de", aspect_spans=spans)

    assert weighted == ["Login fails"]


# ── embedder prefixes ────────────────────────────────────────────────────────

def test_the_embedder_uses_the_configured_prefixes(monkeypatch):
    """BGE-M3 wants no prefix; the E5 "passage: " used to be hard-coded."""
    import src.retrieval.retriever as retriever_module

    seen = {}

    class FakeSentenceTransformer:
        max_seq_length = 128

        def __init__(self, name, device=None):
            seen["name"] = name

        def encode(self, texts, **kwargs):
            seen["texts"] = texts
            return np.zeros((len(texts), 3), dtype=np.float32)

    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers",
                        type("M", (), {"SentenceTransformer":
                                       FakeSentenceTransformer,
                                       "CrossEncoder": object}))

    config = StubConfig(query_prefix="query: ", passage_prefix="passage: ",
                        max_length=256)
    embedder = retriever_module.STEmbedder(config)

    assert embedder.model.max_seq_length == 256
    embedder.encode(["hello"], kind="query")
    assert seen["texts"] == ["query: hello"]
    embedder.encode(["hello"], kind="passage")
    assert seen["texts"] == ["passage: hello"]

    bare = retriever_module.STEmbedder(StubConfig())
    bare.encode(["hello"], kind="query")
    assert seen["texts"] == ["hello"]


# ── indexer side ─────────────────────────────────────────────────────────────

def test_the_indexer_stores_body_and_the_gold_columns(tmp_path):
    from src.retrieval.indexer import CorpusIndexer, IndexerConfig

    config = IndexerConfig(
        aspects_csv="", train_csv="", col_ticket_id="ticket_id",
        col_subject="subject", col_body="body", col_response="answer",
        col_language="language", col_aspects=["prob_sub"],
        index_path="", metadata_path="", dense_path="", model_name="stub",
        device="cpu", batch_size=4, max_length=512, body_max_chars=0,
        metadata_body_max_chars=10)
    indexer = CorpusIndexer.__new__(CorpusIndexer)
    indexer.config = config

    payload = indexer._metadata_payload({
        "ticket_id": "t9", "subject": "S", "body": "0123456789abcdef",
        "response": "R", "language": "de",
        "aspects": {"prob_sub": "p", "queue": "Billing"}})

    assert payload["body"] == "0123456789"
    assert payload["aspects"] == {"prob_sub": "p", "queue": "Billing"}


def test_the_indexer_refuses_an_index_containing_evaluation_rows(tmp_path):
    import pandas as pd

    from src.retrieval.indexer import CorpusIndexer, IndexerConfig

    sample = tmp_path / "test_500.csv"
    pd.DataFrame([{"idx": "t1"}]).to_csv(sample, index=False)
    config = IndexerConfig(
        aspects_csv="", train_csv="", col_ticket_id="ticket_id",
        col_subject="subject", col_body="body", col_response="answer",
        col_language="language", col_aspects=[], index_path="",
        metadata_path="", dense_path="", model_name="stub", device="cpu",
        batch_size=4, max_length=512, body_max_chars=0,
        exclude_idx_csvs=[str(sample)])
    indexer = CorpusIndexer.__new__(CorpusIndexer)
    indexer.config = config

    with pytest.raises(AssertionError, match="evaluation rows"):
        indexer.assert_no_evaluation_rows([{"ticket_id": "t1"},
                                           {"ticket_id": "t2"}])

    # no overlap -> no complaint
    indexer.assert_no_evaluation_rows([{"ticket_id": "t3"}])


def test_records_without_aspects_still_carry_the_gold_columns():
    import pandas as pd

    from src.retrieval.indexer import CorpusIndexer, IndexerConfig

    config = IndexerConfig(
        aspects_csv="", train_csv="", col_ticket_id="ticket_id",
        col_subject="subject", col_body="body", col_response="answer",
        col_language="language", col_aspects=["prob_sub"], index_path="",
        metadata_path="", dense_path="", model_name="stub", device="cpu",
        batch_size=4, max_length=512, body_max_chars=0)
    indexer = CorpusIndexer.__new__(CorpusIndexer)
    indexer.config = config
    df = pd.DataFrame([{"ticket_id": "t1", "subject": "S", "body": "B",
                        "answer": "A", "language": "en", "queue": "Billing",
                        "type": "Request", "priority": "low",
                        "prob_sub": "ignored"}])

    records = indexer._records_from(df, with_aspects=False)

    assert records[0]["aspects"] == {"queue": "Billing", "type": "Request",
                                     "priority": "low"}
