"""Step 7 (plan §6.4): retrieval-only proxy metrics."""

import numpy as np
import pandas as pd
import pytest

from src.retrieval.eval_retrieval import (METRICS, bootstrap_mean, cosine,
                                          evaluate_rows, intent_at_k, rouge_l,
                                          summarise)


# ── per-row metrics ──────────────────────────────────────────────────────────

def test_top1_rouge_uses_the_unicode_tokenizer():
    assert rouge_l("Größe ändern", "Größe ändern") == 1.0
    assert rouge_l("völlig anderes", "Größe ändern") == 0.0
    assert 0.0 < rouge_l("Die Größe wurde geändert", "Größe ändern") < 1.0


def test_rouge_handles_an_empty_retrieval():
    assert rouge_l("", "Größe ändern") == 0.0


class _StubEmbedder:
    """Embeds by token-set, so cosine is predictable without a model."""

    VOCAB = ["login", "password", "invoice", "refund"]

    def encode(self, texts, kind="passage"):
        vectors = []
        for text in texts:
            tokens = set((text or "").lower().split())
            vectors.append([1.0 if word in tokens else 0.0
                            for word in self.VOCAB])
        return np.asarray(vectors, dtype=np.float32)


def test_cosine_is_one_for_identical_text_and_zero_for_disjoint():
    embedder = _StubEmbedder()

    assert cosine(embedder, "login password", "login password") == pytest.approx(1.0)
    assert cosine(embedder, "login", "invoice") == pytest.approx(0.0)
    assert cosine(embedder, "", "login") == 0.0


def test_intent_at_3_is_the_share_of_matching_hits():
    hits = [{"aspects": {"intent": "Billing"}},
            {"aspects": {"queue": "billing"}},
            {"aspects": {"intent": "Technical Support"}}]

    assert intent_at_k("billing", hits) == pytest.approx(2 / 3)


def test_intent_at_3_matches_a_predicted_intent_against_gold_queue():
    hits = [{"aspects": {"queue": "Technical Support"}}]

    assert intent_at_k("technical support", hits) == 1.0


def test_intent_at_3_is_none_when_it_cannot_be_judged():
    """A row that cannot be scored is excluded from the mean, not counted as 0."""
    assert intent_at_k("", [{"aspects": {"intent": "billing"}}]) is None
    assert intent_at_k("none", [{"aspects": {"intent": "billing"}}]) is None
    assert intent_at_k("billing", []) is None
    assert intent_at_k("billing", [{"aspects": {}}]) is None


# ── bootstrap ────────────────────────────────────────────────────────────────

def test_bootstrap_reports_a_mean_and_a_ci_around_it():
    stats = bootstrap_mean([0.1, 0.2, 0.3, 0.4, 0.5])

    assert stats["mean"] == pytest.approx(0.3)
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]
    assert stats["n"] == 5


def test_bootstrap_is_reproducible():
    values = [0.1, 0.9, 0.4, 0.2, 0.7]

    assert bootstrap_mean(values) == bootstrap_mean(values)


def test_bootstrap_skips_missing_values():
    stats = bootstrap_mean([1.0, None, float("nan"), 0.0])

    assert stats["n"] == 2
    assert stats["mean"] == pytest.approx(0.5)


def test_bootstrap_of_nothing_is_not_a_crash():
    stats = bootstrap_mean([])

    assert stats["n"] == 0
    assert np.isnan(stats["mean"])


def test_a_constant_metric_has_a_degenerate_ci():
    stats = bootstrap_mean([0.5] * 10)

    assert (stats["mean"], stats["ci_low"], stats["ci_high"]) == (0.5, 0.5, 0.5)


# ── evaluate_rows / summarise ────────────────────────────────────────────────

class _StubRetriever:
    def __init__(self):
        self.embedder = _StubEmbedder()
        self.calls = []

    def retrieve(self, subject, body, language, aspect_spans=None, top_k=3,
                 exclude_ids=None):
        self.calls.append({"language": language, "exclude_ids": exclude_ids,
                           "aspect_spans": aspect_spans, "top_k": top_k})
        if language.startswith("de"):
            return [{"response": "Passwort zurücksetzen", "ce_score": 0.7,
                     "aspects": {"queue": "Technical Support"}}]
        return [{"response": "Reset your password", "ce_score": 0.9,
                 "aspects": {"intent": "Technical Support"}},
                {"response": "Check the invoice", "ce_score": 0.4,
                 "aspects": {"intent": "Billing"}}]


@pytest.fixture
def val_rows():
    return pd.DataFrame([
        {"idx": "1", "language": "en", "subject": "Login fails",
         "body": "cannot log in", "answer": "Reset your password",
         "intent": "Technical Support", "prob_sub": "login fails",
         "cause": "none"},
        {"idx": "2", "language": "de", "subject": "Anmeldung",
         "body": "komme nicht rein", "answer": "Passwort zurücksetzen",
         "intent": "Technical Support", "prob_sub": "anmeldung",
         "cause": "none"},
    ])


def test_evaluate_rows_scores_every_row(val_rows):
    per_row = evaluate_rows(_StubRetriever(), val_rows, top_k=3)

    assert list(per_row["idx"]) == ["1", "2"]
    assert list(per_row["language"]) == ["en", "de"]
    # top-1 response equals the gold answer in both rows
    assert list(per_row["top1_rougeL"]) == [1.0, 1.0]
    assert per_row["top1_cos"].iloc[0] == pytest.approx(1.0)
    assert per_row["intent@3"].iloc[0] == pytest.approx(0.5)
    assert list(per_row["top1_ce"]) == [0.9, 0.7]


def test_evaluate_rows_excludes_the_query_row_from_its_own_results(val_rows):
    retriever = _StubRetriever()

    evaluate_rows(retriever, val_rows)

    assert [call["exclude_ids"] for call in retriever.calls] == [{"1"}, {"2"}]


def test_evaluate_rows_passes_the_aspect_spans(val_rows):
    retriever = _StubRetriever()

    evaluate_rows(retriever, val_rows)

    spans = retriever.calls[0]["aspect_spans"]
    assert spans["intent"] == "Technical Support"
    assert "cause" not in spans          # 'none' is dropped


def test_evaluate_rows_handles_a_row_with_no_hits(val_rows):
    class Empty(_StubRetriever):
        def retrieve(self, *args, **kwargs):
            return []

    per_row = evaluate_rows(Empty(), val_rows)

    assert list(per_row["n_hits"]) == [0, 0]
    assert list(per_row["top1_rougeL"]) == [0.0, 0.0]
    assert per_row["intent@3"].isna().all()


def test_summarise_reports_every_metric_overall_and_per_language(val_rows):
    per_row = evaluate_rows(_StubRetriever(), val_rows)

    table = summarise(per_row, "m3_hybrid_rerank")

    assert set(table["split"]) == {"all", "en", "de"}
    assert set(table["metric"]) == set(METRICS)
    assert set(table["config"]) == {"m3_hybrid_rerank"}
    for column in ("mean", "ci_low", "ci_high", "n"):
        assert column in table.columns


def test_the_table_keeps_other_configs_and_replaces_its_own(tmp_path,
                                                            monkeypatch,
                                                            val_rows):
    import src.retrieval.eval_retrieval as module

    input_path = tmp_path / "val.csv"
    val_rows.to_csv(input_path, index=False)
    out_path = tmp_path / "retrieval_table.csv"

    monkeypatch.setattr(module, "load_retriever_config",
                        lambda *a, **k: object(), raising=False)
    monkeypatch.setattr("src.retrieval.inference_retriever.load_retriever_config",
                        lambda *a, **k: object())
    monkeypatch.setattr("src.retrieval.retriever.TicketRetriever",
                        lambda config: _StubRetriever())

    module.run(str(tmp_path / "cfg.yaml"), str(input_path), "m3_dense",
               str(out_path))
    module.run(str(tmp_path / "cfg.yaml"), str(input_path), "m3_hybrid",
               str(out_path))
    module.run(str(tmp_path / "cfg.yaml"), str(input_path), "m3_dense",
               str(out_path))

    table = pd.read_csv(out_path)
    assert set(table["config"]) == {"m3_dense", "m3_hybrid"}
    # m3_dense was written twice but appears once per (split, metric)
    dense = table[table["config"] == "m3_dense"]
    assert len(dense) == len(dense.drop_duplicates(["split", "metric"]))
