"""Step 8 (plan §7): the Aspect-RAFT build, its leakage guards, and the parts of
`train` that can run without a GPU."""

import json
import random
from pathlib import Path

import pandas as pd
import pytest

from src.step3_raft import (JSONL_FIELDS, RaftConfig, assert_no_evaluation_rows,
                            build_examples, chat_text,
                            drop_self_and_near_duplicates, example_stats,
                            filter_kwargs, latest_checkpoint,
                            metadata_lookup, own_ticket_ids,
                            pick_sanity_examples, random_distractors,
                            read_jsonl, sft_config_kwargs, split_examples,
                            strip_leading_bos, write_jsonl)

METADATA = [
    {"ticket_id": "t0", "subject": "Login fails", "body": "I cannot log in",
     "response": "Reset your password", "language": "en",
     "aspects": {"queue": "Technical Support"}},
    {"ticket_id": "t1", "subject": "Invoice wrong", "body": "charged twice",
     "response": "We refunded it", "language": "en",
     "aspects": {"queue": "Billing"}},
    {"ticket_id": "t2", "subject": "Anmeldung", "body": "komme nicht rein",
     "response": "Passwort zurücksetzen", "language": "de",
     "aspects": {"queue": "Technical Support"}},
]


# ── own-row exclusion ────────────────────────────────────────────────────────

def test_a_training_row_finds_its_own_ticket_id():
    """Without this the row retrieves itself and the model learns to copy."""
    lookup = metadata_lookup(METADATA)

    assert own_ticket_ids(lookup, "Login fails", "I cannot log in", "en") == {"t0"}
    # the body is matched on its first 200 characters
    assert own_ticket_ids(lookup, "Login fails", "I cannot log in today",
                          "en") == set()
    assert own_ticket_ids(lookup, "Unknown", "x", "en") == set()


def test_the_lookup_keeps_the_first_id_for_duplicate_text():
    lookup = metadata_lookup(METADATA + [dict(METADATA[0], ticket_id="t9")])

    assert lookup[("Login fails", "I cannot log in", "en")] == "t0"


# ── near-duplicate filter ────────────────────────────────────────────────────

def test_a_near_duplicate_hit_is_dropped():
    hits = [{"dense": 0.99, "response": "something else"},
            {"dense": 0.70, "response": "a real past answer"}]

    kept = drop_self_and_near_duplicates(hits, "gold", 0.97, 3)

    assert [h["dense"] for h in kept] == [0.70]


def test_a_hit_whose_response_is_the_gold_answer_is_dropped():
    hits = [{"dense": 0.5, "response": "  Reset your password "},
            {"dense": 0.4, "response": "Other"}]

    kept = drop_self_and_near_duplicates(hits, "Reset your password", 0.97, 3)

    assert [h["response"] for h in kept] == ["Other"]


def test_the_filter_keeps_at_most_top_k():
    hits = [{"dense": 0.5, "response": f"r{i}"} for i in range(10)]

    assert len(drop_self_and_near_duplicates(hits, "gold", 0.97, 3)) == 3


def test_a_hit_without_a_dense_score_survives():
    hits = [{"response": "r", "ce_score": 0.9}]

    assert drop_self_and_near_duplicates(hits, "gold", 0.97, 3) == hits


# ── distractors ──────────────────────────────────────────────────────────────

def test_distractors_prefer_a_different_intent():
    rng = random.Random(42)

    picked = random_distractors(METADATA, rng, 2,
                                avoid_intent="Technical Support")

    assert {p["ticket_id"] for p in picked} == {"t1"}


def test_distractors_fall_back_to_the_whole_pool():
    rng = random.Random(42)

    picked = random_distractors(METADATA, rng, 3, avoid_intent="")

    assert len(picked) == 3
    assert all(p["dense"] == 0.0 for p in picked)


def test_distractors_of_an_empty_index_are_empty():
    assert random_distractors([], random.Random(1), 3) == []


# ── leakage assertion ────────────────────────────────────────────────────────

def test_training_on_an_evaluation_row_is_refused(tmp_path):
    """plan §7.4: the test and validate samples are never trained on."""
    evaluation = tmp_path / "test_500.csv"
    pd.DataFrame([{"idx": "5"}]).to_csv(evaluation, index=False)
    train = pd.DataFrame([{"idx": "5"}, {"idx": "6"}])

    with pytest.raises(AssertionError, match="train split only"):
        assert_no_evaluation_rows(train, [str(evaluation)])

    assert_no_evaluation_rows(pd.DataFrame([{"idx": "7"}]), [str(evaluation)])


# ── build_examples ───────────────────────────────────────────────────────────

class _StubRetriever:
    def __init__(self):
        self.metadata = METADATA
        self.calls = []

    def retrieve(self, subject, body, language, aspect_spans=None, top_k=3,
                 exclude_ids=None):
        self.calls.append({"exclude_ids": exclude_ids, "top_k": top_k,
                           "aspect_spans": aspect_spans})
        return [{"subject": "Past", "body": "past body",
                 "response": "a real past answer", "dense": 0.5,
                 "aspects": {}, "ticket_id": "tX"}]

    def build_prompt(self, subject, body, retrieved, aspect_spans=None,
                     language="en"):
        aspects = "with-aspects" if aspect_spans else "no-aspects"
        return (f"{aspects} ctx={len(retrieved)} {subject} [{language}] "
                f"Support reply:")


def _words(text):
    return len(text.split())


@pytest.fixture
def train_rows():
    return pd.DataFrame([
        {"idx": "100", "language": "en", "subject": "Login fails",
         "body": "I cannot log in", "answer": "Reset your password",
         "prob_sub": "login fails", "cause": "none", "intent": "Technical Support",
         "ticket_type_nli": "Incident", "action_taken": "none", "entities": "",
         "urgency": "high", "prob_statement": "login fails"},
        {"idx": "101", "language": "de", "subject": "Anmeldung",
         "body": "komme nicht rein", "answer": "Passwort zurücksetzen",
         "prob_sub": "anmeldung", "cause": "none", "intent": "Technical Support",
         "ticket_type_nli": "Incident", "action_taken": "none", "entities": "",
         "urgency": "low", "prob_statement": "anmeldung"},
    ])


ASPECTS = ["prob_sub", "prob_statement", "cause", "intent", "ticket_type_nli",
           "entities", "action_taken", "urgency"]


def test_build_writes_one_example_per_row(train_rows):
    config = RaftConfig(distractor_prob=0.0, use_aspects_prob=1.0,
                        max_seq_len=4096)

    examples = build_examples(train_rows, _StubRetriever(), _words, config,
                              ASPECTS)

    assert [e["idx"] for e in examples] == ["100", "101"]
    assert [e["language"] for e in examples] == ["en", "de"]
    assert all(set(e) == set(JSONL_FIELDS) for e in examples)
    assert all(e["prompt"].endswith("Support reply:") for e in examples)
    assert all(e["with_aspects"] for e in examples)
    assert not any(e["distractor"] for e in examples)


def test_build_excludes_each_rows_own_ticket(train_rows):
    retriever = _StubRetriever()
    config = RaftConfig(distractor_prob=0.0, use_aspects_prob=1.0)

    build_examples(train_rows, retriever, _words, config, ASPECTS)

    assert retriever.calls[0]["exclude_ids"] == {"t0"}
    assert retriever.calls[1]["exclude_ids"] == {"t2"}


def test_build_asks_for_extra_candidates_to_survive_the_filter(train_rows):
    retriever = _StubRetriever()
    config = RaftConfig(top_k=3, distractor_prob=0.0)

    build_examples(train_rows, retriever, _words, config, ASPECTS)

    assert all(call["top_k"] == 6 for call in retriever.calls)


def test_build_skips_rows_without_a_gold_answer(train_rows):
    train_rows.loc[0, "answer"] = ""

    examples = build_examples(train_rows, _StubRetriever(), _words,
                              RaftConfig(), ASPECTS)

    assert [e["idx"] for e in examples] == ["101"]


def test_distractor_prob_one_gives_every_example_random_context(train_rows):
    config = RaftConfig(distractor_prob=1.0, use_aspects_prob=1.0)
    retriever = _StubRetriever()

    examples = build_examples(train_rows, retriever, _words, config, ASPECTS)

    assert all(e["distractor"] for e in examples)
    assert retriever.calls == []        # retrieval is skipped entirely


def test_use_aspects_prob_zero_builds_every_prompt_without_aspects(train_rows):
    """The S5 half of the adapter: 20% of training has no aspect checklist."""
    config = RaftConfig(distractor_prob=0.0, use_aspects_prob=0.0)

    examples = build_examples(train_rows, _StubRetriever(), _words, config,
                              ASPECTS)

    assert not any(e["with_aspects"] for e in examples)
    assert all("no-aspects" in e["prompt"] for e in examples)


def test_build_is_reproducible_for_a_seed(train_rows):
    config = RaftConfig(distractor_prob=0.5, use_aspects_prob=0.5, seed=42)

    first = build_examples(train_rows, _StubRetriever(), _words, config, ASPECTS)
    second = build_examples(train_rows, _StubRetriever(), _words, config,
                            ASPECTS)

    assert first == second


def test_the_prompt_budget_applies_to_training_prompts_too(train_rows):
    """The point of RAFT: the training prompt is the inference prompt, budget
    included — a long past answer is dropped rather than truncated."""

    class LongContext(_StubRetriever):
        def retrieve(self, *args, **kwargs):
            return [{"subject": "Past", "body": "b", "dense": 0.5,
                     "response": "word " * 500, "aspects": {},
                     "ticket_id": "tX"}]

        def build_prompt(self, subject, body, retrieved, aspect_spans=None,
                         language="en"):
            context = " ".join(r["response"] for r in retrieved)
            return f"ctx={len(retrieved)} {context} {subject} Support reply:"

    config = RaftConfig(distractor_prob=0.0, max_seq_len=512)   # budget 256

    examples = build_examples(train_rows, LongContext(), _words, config,
                              ASPECTS)

    assert all("ctx=0" in e["prompt"] for e in examples)
    assert all(_words(e["prompt"]) + 16 <= 256 for e in examples)
    assert all(e["prompt"].endswith("Support reply:") for e in examples)


# ── stats and JSONL ──────────────────────────────────────────────────────────

def test_stats_report_the_shares_the_plan_asks_for():
    examples = [
        {"language": "en", "distractor": True, "with_aspects": True,
         "prompt": "a b c"},
        {"language": "de", "distractor": False, "with_aspects": True,
         "prompt": "a b"},
        {"language": "de", "distractor": False, "with_aspects": False,
         "prompt": "a"},
        {"language": "en", "distractor": False, "with_aspects": True,
         "prompt": "a b c d"},
    ]

    stats = example_stats(examples, _words)

    assert stats["n"] == 4
    assert (stats["en"], stats["de"]) == (2, 2)
    assert stats["distractor_share"] == 0.25
    assert stats["with_aspects_share"] == 0.75
    assert stats["prompt_tokens_p50"] == 2
    assert stats["prompt_tokens_p95"] >= stats["prompt_tokens_p50"]


def test_stats_of_nothing_do_not_divide_by_zero():
    assert example_stats([])["n"] == 0


def test_jsonl_round_trips_with_german_intact(tmp_path):
    examples = [{"idx": "1", "language": "de", "prompt": "Größe ändern",
                 "answer": "Vielen Dank", "distractor": False,
                 "with_aspects": True}]
    path = tmp_path / "raft.jsonl"

    write_jsonl(examples, str(path))

    assert read_jsonl(str(path)) == examples
    assert "Größe ändern" in path.read_text(encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8").splitlines()[0])


# ── train helpers ────────────────────────────────────────────────────────────

def test_a_leading_bos_is_stripped():
    """The trainer adds <bos>; two of them shift every position."""
    assert strip_leading_bos("<bos>hello") == "hello"
    assert strip_leading_bos("hello") == "hello"


class _StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False):
        if not isinstance(messages[0]["content"], str):
            raise TypeError("this template wants plain strings")
        return "<bos>" + "|".join(f"{m['role']}:{m['content']}"
                                  for m in messages)


def test_chat_text_contains_the_prompt_and_the_answer_without_a_second_bos():
    text = chat_text(_StubTokenizer(), "the prompt", "the answer")

    assert not text.startswith("<bos>")
    assert "user:the prompt" in text
    assert "assistant:the answer" in text


def test_chat_text_falls_back_to_typed_content():
    class TypedOnly:
        def apply_chat_template(self, messages, tokenize=False):
            if isinstance(messages[0]["content"], str):
                raise TypeError("typed content only")
            return "<bos>" + messages[0]["content"][0]["text"]

    assert chat_text(TypedOnly(), "the prompt", "a") == "the prompt"


def test_the_validation_split_is_held_out_and_reproducible():
    examples = [{"idx": str(i)} for i in range(100)]

    train, val = split_examples(examples, 0.05, 42)
    train_again, val_again = split_examples(examples, 0.05, 42)

    assert len(val) == 5
    assert len(train) == 95
    assert not {e["idx"] for e in train} & {e["idx"] for e in val}
    assert val == val_again and train == train_again


def test_a_zero_validation_fraction_leaves_no_eval_set():
    train, val = split_examples([{"idx": "1"}, {"idx": "2"}], 0.0, 42)

    assert val == []
    assert len(train) == 2


def test_a_tiny_dataset_still_keeps_one_validation_example():
    train, val = split_examples([{"idx": "1"}, {"idx": "2"}], 0.05, 42)

    assert len(val) == 1 and len(train) == 1


def test_the_latest_checkpoint_is_found_for_a_resumed_session(tmp_path):
    for step in (50, 150, 100):
        (tmp_path / f"checkpoint-{step}").mkdir()

    assert latest_checkpoint(str(tmp_path)).endswith("checkpoint-150")
    assert latest_checkpoint(str(tmp_path / "missing")) is None


def test_sft_arguments_include_eval_only_when_there_is_an_eval_set():
    config = RaftConfig()

    with_eval = sft_config_kwargs(config, has_eval=True)
    without = sft_config_kwargs(config, has_eval=False)

    assert with_eval["eval_strategy"] == "steps"
    assert "eval_strategy" not in without
    assert with_eval["per_device_train_batch_size"] == 2
    assert with_eval["gradient_accumulation_steps"] == 8
    assert with_eval["max_seq_length"] == 2048
    assert sft_config_kwargs(config, False, max_steps=5)["max_steps"] == 5


def test_unknown_trl_arguments_are_dropped_and_renamed():
    """TRL renamed max_seq_length to max_length; the code adapts either way."""
    def new_trl(dataset_text_field=None, max_length=None, seed=None):
        pass

    def old_trl(dataset_text_field=None, max_seq_length=None, seed=None):
        pass

    kwargs = {"dataset_text_field": "text", "max_seq_length": 2048, "seed": 42,
              "report_to": "none"}

    assert filter_kwargs(new_trl, kwargs) == {"dataset_text_field": "text",
                                              "max_length": 2048, "seed": 42}
    assert filter_kwargs(old_trl, kwargs) == {"dataset_text_field": "text",
                                              "max_seq_length": 2048,
                                              "seed": 42}


def test_the_sanity_check_picks_one_en_one_de_and_one_distractor():
    examples = [
        {"language": "en", "distractor": False, "idx": "1"},
        {"language": "en", "distractor": False, "idx": "2"},
        {"language": "de", "distractor": False, "idx": "3"},
        {"language": "en", "distractor": True, "idx": "4"},
    ]

    picked = pick_sanity_examples(examples)

    assert [e["idx"] for e in picked] == ["1", "3", "4"]


def test_the_sanity_check_copes_with_a_one_language_dataset():
    picked = pick_sanity_examples([{"language": "en", "distractor": False,
                                    "idx": "1"}])

    assert [e["idx"] for e in picked] == ["1"]


# ── config ───────────────────────────────────────────────────────────────────

def test_the_raft_config_loads_and_the_kaggle_overrides_apply():
    from src.step3_raft import load_raft_config

    local = load_raft_config("config/config_raft.yaml")
    kaggle = load_raft_config("config/config_raft.yaml", "config/kaggle.yaml")

    assert local.base_model == "unsloth/gemma-3-4b-it"
    assert local.top_k == 3
    assert local.distractor_prob == 0.2
    assert local.use_aspects_prob == 0.8
    assert local.lora["r"] == 16
    assert local.hf_repo == "ranjankn/aspectforge-raft-gemma3"
    # config/kaggle.yaml's raft section
    assert kaggle.max_seq_len == 2048
    assert kaggle.batch == 2
    assert kaggle.grad_accum == 8
    assert kaggle.out_dir.startswith("/kaggle/working/Code")
    assert kaggle.raft_jsonl.startswith("/kaggle/working/Code")


def test_the_raft_config_keeps_relative_paths_locally():
    from src.step3_raft import load_raft_config

    config = load_raft_config("config/config_raft.yaml")

    assert config.raft_jsonl == "output/raft/raft_train.jsonl"
    assert "/kaggle/" not in Path("config/config_raft.yaml").read_text(
        encoding="utf-8")
