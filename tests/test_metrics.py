"""Step 2 (plan §8.1): Unicode ROUGE, real sacreBLEU, true BERTScore,
lang_match, and a per-row form for every metric."""

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.utils import MetricsCalculator, _UnicodeTokenizer

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "docs" / "samples"


@dataclass
class MetCfg:
    rouge_types      : list = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
    rouge_use_stemmer: bool = True
    rouge_unicode_tokenizer: bool = True
    bleu_tokenizer   : str  = "13a"
    bertscore_true_model: str = "xlm-roberta-large"
    bertscore_true_batch: int = 16


@pytest.fixture
def calc():
    return MetricsCalculator(MetCfg())


# ── tokenizer ────────────────────────────────────────────────────────────────

def test_unicode_tokenizer_keeps_umlauts_and_lowercases():
    assert _UnicodeTokenizer().tokenize("Größe ändern, bitte!") == ["größe", "ändern", "bitte"]
    assert _UnicodeTokenizer().tokenize(None) == []


# ── ROUGE ────────────────────────────────────────────────────────────────────

def test_rouge_scores_identical_german_as_one(calc):
    """The plan's acceptance test: ROUGE must not be destroyed by umlauts."""
    scores = calc.rouge(["Größe ändern"], ["Größe ändern"])

    assert scores["rougeL"] == 1.0
    assert scores["rouge1"] == 1.0


def test_rouge_without_the_unicode_tokenizer_loses_german(calc):
    """The old ASCII behaviour stays reachable via config — and is still wrong:
    it strips umlauts, so two different German words score as identical."""
    assert calc.rouge(["Größe"], ["Grüße"])["rougeL"] == 0.0

    calc.cfg.rouge_unicode_tokenizer = False

    assert calc.rouge(["Größe"], ["Grüße"])["rougeL"] == 1.0


def test_rouge_per_row_lists_align_with_the_inputs(calc):
    scores = calc.rouge(["Größe ändern", "völlig falsch"],
                        ["Größe ändern", "etwas anderes"], per_row=True)

    assert scores["rougeL_rows"][0] == 1.0
    assert scores["rougeL_rows"][1] == 0.0
    assert len(scores["rouge1_rows"]) == 2
    assert scores["rougeL"] == pytest.approx(0.5)


def test_rouge_is_not_perfect_for_a_partial_match(calc):
    scores = calc.rouge(["Die Größe wurde geändert"], ["Größe ändern"])

    assert 0.0 < scores["rougeL"] < 1.0


# ── BLEU ─────────────────────────────────────────────────────────────────────

def test_bleu_reports_real_sacrebleu_and_keeps_the_precisions(calc):
    long_text = ("Vielen Dank für Ihre Nachricht, wir haben die Größe "
                 "Ihres Kontos geändert und melden uns in Kürze.")
    scores = calc.bleu([long_text], [long_text])

    assert set(scores) == {"bleu", "bleu_p1", "bleu_p4"}
    assert scores["bleu"] == pytest.approx(100.0, abs=0.01)   # 0-100 scale
    assert scores["bleu_p1"] == 1.0
    assert scores["bleu_p4"] == 1.0


def test_bleu_drops_for_an_unrelated_hypothesis(calc):
    scores = calc.bleu(["completely unrelated words here"],
                       ["Vielen Dank für Ihre Nachricht"])

    assert scores["bleu"] < 1.0


# ── lang_match ───────────────────────────────────────────────────────────────

def test_lang_match_is_one_when_every_reply_uses_the_ticket_language(calc):
    """The plan's acceptance test."""
    scores = calc.lang_match(["Hello there, thanks", "Vielen Dank für Ihre Nachricht"],
                             ["en", "de"])

    assert scores["lang_match"] == 1.0


def test_lang_match_catches_a_reply_in_the_wrong_language(calc):
    scores = calc.lang_match(["Hello there, thanks for reaching out to us",
                              "Thank you for your message, we will look into it"],
                             ["en", "de"], per_row=True)

    assert scores["lang_match_rows"] == [1, 0]
    assert scores["lang_match"] == 0.5


def test_lang_match_counts_undetectable_text_as_a_miss(calc):
    scores = calc.lang_match(["", "   "], ["en", "de"], per_row=True)

    assert scores["lang_match_rows"] == [0, 0]
    assert scores["lang_match"] == 0.0


def test_lang_match_accepts_locale_style_language_codes(calc):
    scores = calc.lang_match(["Vielen Dank für Ihre Nachricht"], ["de-DE"])

    assert scores["lang_match"] == 1.0


def test_lang_match_is_deterministic(calc):
    text = ["Die Rechnung wurde korrigiert und erneut versandt."]

    assert calc.lang_match(text, ["de"]) == calc.lang_match(text, ["de"])


# ── true BERTScore ───────────────────────────────────────────────────────────

def test_bertscore_true_uses_bert_score_and_returns_per_row_f1(calc, monkeypatch):
    """xlm-roberta-large is a 2 GB download, so the call itself is stubbed; what
    matters here is the key names, the defaults and the per-row list."""
    seen = {}

    def fake_score(hyps, refs, model_type=None, batch_size=None, verbose=False):
        seen.update(model_type=model_type, batch_size=batch_size, n=len(hyps))
        return None, None, _FakeTensor([0.9, 0.7])

    monkeypatch.setitem(sys.modules, "bert_score",
                        types.SimpleNamespace(score=fake_score))

    scores = calc.bertscore_true(["a", "b"], ["a", "c"], per_row=True)

    assert seen == {"model_type": "xlm-roberta-large", "batch_size": 16, "n": 2}
    assert scores["bertscore_f1"] == 0.8
    assert scores["bertscore_f1_rows"] == [0.9, 0.7]


def test_bertscore_true_honours_explicit_arguments(calc, monkeypatch):
    seen = {}

    def fake_score(hyps, refs, model_type=None, batch_size=None, verbose=False):
        seen.update(model_type=model_type, batch_size=batch_size)
        return None, None, _FakeTensor([1.0])

    monkeypatch.setitem(sys.modules, "bert_score",
                        types.SimpleNamespace(score=fake_score))
    calc.bertscore_true(["a"], ["a"], model_type="bert-base-multilingual-cased",
                        batch_size=4)

    assert seen == {"model_type": "bert-base-multilingual-cased", "batch_size": 4}


class _FakeTensor:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


# ── the CLS cosine keeps its own name ────────────────────────────────────────

def test_cls_cosine_is_no_longer_called_bertscore(monkeypatch):
    """`bertscore()` is a CLS cosine, not BERTScore: plan §8.1 renames its output
    to cls_cosine_mbert so the report cannot confuse the two."""
    import inspect

    import src.utils as utils

    source = inspect.getsource(utils.MetricsCalculator.bertscore)

    assert "cls_cosine_mbert" in source
    assert '"bertscore_f1"' not in source


# ── on the real sample rows ──────────────────────────────────────────────────

def test_metrics_run_on_the_committed_sample_rows(calc):
    import pandas as pd

    df = pd.read_csv(SAMPLES / "cs-support-multi-test_rag_only_SAMPLE_OUTPUT_RAG_ONLY.csv",
                     dtype=str).fillna("")
    reply_col = next(c for c in df.columns if c.endswith("_response"))
    hyps = df[reply_col].tolist()
    refs = df["answer"].tolist()

    rouge = calc.rouge(hyps, refs, per_row=True)
    bleu = calc.bleu(hyps, refs)
    langs = calc.lang_match(hyps, df["language"].tolist(), per_row=True)

    assert len(rouge["rougeL_rows"]) == len(df)
    assert 0.0 <= rouge["rougeL"] <= 1.0
    assert bleu["bleu"] >= 0.0
    assert len(langs["lang_match_rows"]) == len(df)
    assert set(df["language"].str.lower().str[:2]) <= {"en", "de"}
