"""Step 3 (plan §5.2–5.5): model_backend, prompt budget, reply-language line and
the German sentiment/severity fix. The GPU paths are smoke-tested on Kaggle; what
is checked here is the plumbing around them."""

import types

import pandas as pd
import pytest

from src.generation.model_backend import (BACKENDS, count_tokens_fn,
                                          eos_token_ids, make_chat_fn,
                                          text_tokenizer)
from src.retrieval.retriever import LANG_NAMES, reply_language_line
from src.utils import fit_prompt


# ── §5.3 prompt budget ───────────────────────────────────────────────────────

def _words(text):
    """A token count that is cheap and monotone in length."""
    return len(text.split())


def test_fit_prompt_keeps_every_doc_when_the_budget_allows():
    docs = ["a", "b", "c"]

    prompt, used = fit_prompt(lambda d: " ".join(d), docs, _words, 100)

    assert prompt == "a b c"
    assert used == 3


def test_fit_prompt_drops_the_lowest_ranked_docs_until_it_fits():
    docs = ["one", "two", "three", "four"]

    prompt, used = fit_prompt(lambda d: " ".join(["head"] + d), docs, _words, 19)

    # head + 2 docs = 3 words, +16 headroom = 19
    assert used == 2
    assert prompt == "head one two"


def test_fit_prompt_returns_the_bare_prompt_when_nothing_fits():
    """The instruction and the ticket are never dropped — only context is."""
    def build(docs):
        return "instruction " + " ".join(docs) + " Support reply:"

    prompt, used = fit_prompt(build, ["x", "y"], _words, 1)

    assert used == 0
    assert prompt.endswith("Support reply:")


def test_fit_prompt_accepts_no_retrieved_docs():
    prompt, used = fit_prompt(lambda d: "only the ticket", [], _words, 50)

    assert (prompt, used) == ("only the ticket", 0)


def test_the_real_prompt_stays_within_budget_with_five_long_hits():
    """Plan §5.3's acceptance case: long hits must not push the prompt over the
    limit, and the prompt must still end with "Support reply:"."""
    retriever = _stub_retriever(ctx_doc_max_chars=1200)
    hits = [{"subject": f"Ticket {i}", "response": "word " * 800} for i in range(5)]
    spans = {"prob_sub": "The export fails", "urgency": "high"}
    max_input_tokens = 512

    prompt, used = fit_prompt(
        lambda docs: retriever.build_prompt("Subj", "Body " * 50, docs, spans, "en"),
        hits, _words, max_input_tokens)

    assert _words(prompt) + 16 <= max_input_tokens
    assert used < len(hits)
    assert prompt.rstrip().endswith("Support reply:")


# ── §5.3 per-document cap ────────────────────────────────────────────────────

def _stub_retriever(**overrides):
    from src.retrieval.retriever import TicketRetriever

    config = types.SimpleNamespace(aspect_prompt_style="checklist",
                                   prompt_body_max_chars=0,
                                   ctx_doc_max_chars=0,
                                   reply_language_instruction=True)
    for key, value in overrides.items():
        setattr(config, key, value)
    retriever = TicketRetriever.__new__(TicketRetriever)
    retriever.config = config
    return retriever


def test_ctx_doc_max_chars_caps_each_retrieved_response():
    retriever = _stub_retriever(ctx_doc_max_chars=20)
    hits = [{"subject": "S", "response": "x" * 500}]

    prompt = retriever.build_prompt("Subj", "Body", hits, {"prob_sub": "p"}, "en")

    assert "x" * 20 in prompt
    assert "x" * 21 not in prompt


def test_zero_ctx_doc_max_chars_keeps_the_full_response():
    retriever = _stub_retriever(ctx_doc_max_chars=0)
    hits = [{"subject": "S", "response": "y" * 300}]

    prompt = retriever.build_prompt("Subj", "Body", hits, {"prob_sub": "p"}, "en")

    assert "y" * 300 in prompt


# ── §5.4 reply-language line ─────────────────────────────────────────────────

def test_language_line_names_the_ticket_language():
    assert reply_language_line("de") == ("Write the reply in German, the same "
                                         "language as the customer's message.")
    assert "English" in reply_language_line("en")
    assert "English" in reply_language_line("EN")
    assert "German" in reply_language_line("de-DE")
    assert LANG_NAMES == {"en": "English", "de": "German"}


def test_language_line_falls_back_for_an_unknown_code():
    assert "the customer's language" in reply_language_line("fr")
    # a missing language means English, as everywhere else in the pipeline
    assert "English" in reply_language_line("")


def test_a_german_ticket_gets_a_german_instruction_last():
    retriever = _stub_retriever()
    hits = [{"subject": "S", "response": "R"}]

    prompt = retriever.build_prompt("Betreff", "Nachricht", hits,
                                    {"prob_sub": "Login schlägt fehl"}, "de")

    lines = [line for line in prompt.splitlines() if line.strip()]
    assert "Write the reply in German" in lines[-2]
    assert lines[-1] == "Support reply:"


def test_the_language_line_can_be_switched_off():
    retriever = _stub_retriever(reply_language_instruction=False)

    prompt = retriever.build_prompt("Betreff", "Nachricht",
                                    [{"subject": "S", "response": "R"}],
                                    {"prob_sub": "x"}, "de")

    assert "Write the reply in" not in prompt
    assert prompt.rstrip().endswith("Support reply:")


def test_the_legacy_slot_prompt_also_gets_the_language_line():
    retriever = _stub_retriever(aspect_prompt_style="slots")

    prompt = retriever.build_prompt("Betreff", "Nachricht",
                                    [{"subject": "S", "response": "R"}],
                                    {"prob_sub": "x"}, "de")

    assert "Write the reply in German" in prompt


def test_the_aspect_only_prompt_gets_the_language_line():
    from src.generation.response_generator_v2 import build_aspect_only_prompt

    row = pd.Series({"subject": "Betreff", "body": "Nachricht", "language": "de"})

    prompt = build_aspect_only_prompt(row, {"prob_sub": "Login schlägt fehl"})

    assert "Write the reply in German" in prompt
    assert prompt.rstrip().endswith("Support reply:")


# ── §5.5 sentiment and severity ──────────────────────────────────────────────

def test_sentiment_is_not_invented_for_a_german_ticket():
    from src.generation.response_generator_v2 import extract_sentiment_severity

    sentiment, _, compound = extract_sentiment_severity(
        "Das System ist seit heute Morgen komplett ausgefallen.", "de")

    assert sentiment == "n/a"
    assert compound == 0.0


def test_english_tickets_still_get_a_sentiment():
    from src.generation.response_generator_v2 import extract_sentiment_severity

    sentiment, _, _ = extract_sentiment_severity(
        "This is terrible, I am extremely unhappy with the service.", "en")

    assert sentiment in ("negative", "neutral", "positive")


@pytest.mark.parametrize("body,expected", [
    ("Das ist dringend, wir haben einen Systemausfall.", "critical"),
    ("Es gibt einen Fehler beim Absturz der Anwendung.", "high"),
    ("Die Synchronisation ist sehr langsam.", "medium"),
])
def test_german_severity_keywords_are_recognised(body, expected):
    from src.generation.response_generator_v2 import extract_sentiment_severity

    _, severity, _ = extract_sentiment_severity(body, "de")

    assert severity == expected


def test_the_sentiment_line_is_omitted_when_there_is_no_sentiment():
    from src.generation.response_generator_v2 import inject_sentiment_severity

    with_sentiment = inject_sentiment_severity("head\nbody", "negative",
                                               "high", -0.7)
    without = inject_sentiment_severity("head\nbody", "n/a", "high", 0.0)

    assert "Sentiment: negative" in with_sentiment
    assert "Sentiment" not in without
    assert "Severity : high" in without


# ── §5.2 backend plumbing ────────────────────────────────────────────────────

class _FakeTextTokenizer:
    """Enough of a tokenizer for the helpers: Gemma-3 hides one of these inside
    a processor."""

    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"
    eos_token_id = 1
    unk_token_id = 3
    padding_side = "right"

    def __call__(self, text, add_special_tokens=True, **kwargs):
        texts = [text] if isinstance(text, str) else text
        return {"input_ids": [t.split() for t in texts][0 if isinstance(text, str) else slice(None)]}

    def convert_tokens_to_ids(self, token):
        return 2 if token == "<end_of_turn>" else self.unk_token_id


class _FakeProcessor:
    """A Gemma-3-style processor: the text tokenizer sits behind `.tokenizer`."""

    def __init__(self, typed_content_only=True):
        self.tokenizer = _FakeTextTokenizer()
        self.typed_content_only = typed_content_only

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True):
        content = messages[0]["content"]
        if self.typed_content_only and not isinstance(content, list):
            raise TypeError("this processor wants typed content parts")
        text = content[0]["text"] if isinstance(content, list) else content
        return f"<bos><start_of_turn>user\n{text}<end_of_turn>\n"


def test_text_tokenizer_reaches_through_a_processor():
    processor = _FakeProcessor()

    assert text_tokenizer(processor) is processor.tokenizer
    # a plain tokenizer is returned unchanged
    plain = _FakeTextTokenizer()
    assert text_tokenizer(plain) is plain


def test_chat_fn_uses_typed_content_for_a_gemma_3_processor():
    templated = make_chat_fn(_FakeProcessor(typed_content_only=True))(["hello"])

    assert templated == ["<bos><start_of_turn>user\nhello<end_of_turn>\n"]


def test_chat_fn_falls_back_to_string_content():
    class StringOnly(_FakeProcessor):
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            if isinstance(content, list):
                raise TypeError("plain string content only")
            return f"<bos>{content}"

    assert make_chat_fn(StringOnly())(["hello"]) == ["<bos>hello"]


def test_chat_fn_returns_the_raw_prompt_for_a_pretrained_model():
    class NoTemplate:
        def apply_chat_template(self, *args, **kwargs):
            raise ValueError("no chat template")

    assert make_chat_fn(NoTemplate())(["raw prompt"]) == ["raw prompt"]


def test_eos_includes_gemma_end_of_turn():
    assert eos_token_ids(_FakeProcessor()) == [1, 2]


def test_eos_skips_an_unknown_end_of_turn_token():
    class NoEndOfTurn(_FakeTextTokenizer):
        def convert_tokens_to_ids(self, token):
            return self.unk_token_id

    assert eos_token_ids(NoEndOfTurn()) == [1]


def test_count_tokens_does_not_add_special_tokens():
    seen = {}

    class Counting(_FakeTextTokenizer):
        def __call__(self, text, add_special_tokens=True, **kwargs):
            seen["add_special_tokens"] = add_special_tokens
            return {"input_ids": text.split()}

    count = count_tokens_fn(Counting())

    assert count("one two three") == 3
    assert seen["add_special_tokens"] is False


def test_backends_are_hf_and_unsloth():
    assert BACKENDS == ("hf", "unsloth")


def test_an_unknown_backend_is_rejected():
    from src.generation.model_backend import load_generator

    with pytest.raises(ValueError, match="backend must be one of"):
        load_generator("some/model", backend="vllm")


def test_load_model_rejects_a_seq2seq_model():
    """T5 goes through load_extra_model; the minimal run is decoder-only."""
    from src.generation.response_generator_v2 import (GenerationConfig,
                                                      load_model)

    config = GenerationConfig(model_id="t5-base")

    with pytest.raises(ValueError, match="seq2seq"):
        load_model(config)
