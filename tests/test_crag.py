"""Step 9 (plan §7.5): the CRAG router, its threshold tuning, and reusing S2's
generations for the rows the router sends through RAG."""

import pandas as pd
import pytest

from src.generation.response_generator_v2 import crag_route
from src.minimal_run import (CRAG_THRESHOLDS, reuse_s2_outputs,
                             tune_crag_threshold, write_crag_threshold)


# ── the router ───────────────────────────────────────────────────────────────

def test_a_confident_hit_routes_to_rag():
    assert crag_route([{"ce_score": 0.8}], 0.30) == "rag"
    assert crag_route([{"ce_score": 0.30}], 0.30) == "rag"      # inclusive


def test_a_weak_hit_routes_to_no_context():
    """An irrelevant example is worse than none: the model follows it."""
    assert crag_route([{"ce_score": 0.1}], 0.30) == "no_context"


def test_no_hits_route_to_no_context():
    assert crag_route([], 0.30) == "no_context"


def test_a_hit_without_a_rerank_score_routes_to_no_context():
    """DE used to have no ce_score at all; an unscored hit is not trusted."""
    assert crag_route([{"dense": 0.9}], 0.30) == "no_context"
    assert crag_route([{"ce_score": ""}], 0.30) == "no_context"
    assert crag_route([{"ce_score": None}], 0.30) == "no_context"


def test_only_the_top_hit_decides():
    retrieved = [{"ce_score": 0.1}, {"ce_score": 0.9}]

    assert crag_route(retrieved, 0.30) == "no_context"


# ── threshold tuning ─────────────────────────────────────────────────────────

@pytest.fixture
def paired_rows():
    """Four rows: the two with a high ce_score are answered better with context,
    the two with a low one better without."""
    return pd.DataFrame([
        {"idx": "1", "top_ce_score": "0.90", "answer": "reset your password",
         "rag_response": "reset your password",
         "no_context_response": "please contact support"},
        {"idx": "2", "top_ce_score": "0.50", "answer": "refund issued today",
         "rag_response": "refund issued today",
         "no_context_response": "we will look into it"},
        {"idx": "3", "top_ce_score": "0.10", "answer": "upgrade the plan",
         "rag_response": "unrelated invoice advice",
         "no_context_response": "upgrade the plan"},
        {"idx": "4", "top_ce_score": "0.05", "answer": "clear the cache",
         "rag_response": "unrelated login advice",
         "no_context_response": "clear the cache"},
    ])


def test_tuning_sweeps_the_planned_thresholds(paired_rows):
    table = tune_crag_threshold(paired_rows)

    assert list(table["threshold"]) == CRAG_THRESHOLDS
    assert CRAG_THRESHOLDS[0] == 0.05 and CRAG_THRESHOLDS[-1] == 0.60
    assert set(table.columns) == {"threshold", "rougeL", "rag_share", "n", "best"}
    assert (table["n"] == 4).all()


def test_tuning_finds_the_threshold_that_separates_the_two_groups(paired_rows):
    table = tune_crag_threshold(paired_rows)

    best = float(table.loc[table["best"], "threshold"].iloc[0])
    assert 0.10 < best <= 0.50
    assert table["best"].sum() == 1
    # the best threshold scores better than keeping context everywhere
    assert (table.loc[table["best"], "rougeL"].iloc[0]
            > table.loc[table["threshold"] == 0.05, "rougeL"].iloc[0])


def test_the_rag_share_falls_as_the_threshold_rises(paired_rows):
    table = tune_crag_threshold(paired_rows).set_index("threshold")

    assert table.loc[0.05, "rag_share"] == 1.0
    assert table.loc[0.20, "rag_share"] == 0.5
    # only the 0.90 row still clears the top of the sweep
    assert table.loc[0.60, "rag_share"] == 0.25
    assert table["rag_share"].is_monotonic_decreasing


def test_tuning_treats_a_missing_score_as_no_context(paired_rows):
    paired_rows.loc[0, "top_ce_score"] = ""

    table = tune_crag_threshold(paired_rows).set_index("threshold")

    assert table.loc[0.05, "rag_share"] == 0.75


def test_tuning_needs_both_generations(paired_rows):
    with pytest.raises(ValueError, match="no_context_response"):
        tune_crag_threshold(paired_rows.drop(columns=["no_context_response"]))


def test_tuning_uses_the_unicode_rouge():
    rows = pd.DataFrame([{"idx": "1", "top_ce_score": "0.9",
                          "answer": "Größe ändern",
                          "rag_response": "Größe ändern",
                          "no_context_response": "etwas anderes"}])

    table = tune_crag_threshold(rows, [0.5])

    assert table["rougeL"].iloc[0] == 1.0


def test_the_threshold_is_written_back_into_the_config(tmp_path):
    config = tmp_path / "generation_config.yaml"
    config.write_text("top_k           : 3\ncrag_threshold  : 0.30\n"
                      "output_dir      : out\n", encoding="utf-8")

    write_crag_threshold(str(config), 0.25)

    text = config.read_text(encoding="utf-8")
    assert "crag_threshold  : 0.25" in text
    assert "top_k           : 3" in text
    assert "0.30" not in text


def test_writing_to_a_config_without_the_key_warns_instead_of_corrupting(
        tmp_path, capsys):
    config = tmp_path / "generation_config.yaml"
    config.write_text("top_k: 3\n", encoding="utf-8")

    write_crag_threshold(str(config), 0.25)

    assert config.read_text(encoding="utf-8") == "top_k: 3\n"
    assert "WARNING" in capsys.readouterr().out


# ── reusing S2's generations ─────────────────────────────────────────────────

@pytest.fixture
def s2_frame():
    return pd.DataFrame([
        {"idx": "1", "language": "en", "subject": "a", "answer": "gold 1",
         "top_ce_score": "0.90", "n_ctx_used": "3",
         "s2_gemma3_prompt": "prompt with context 1",
         "s2_gemma3_response": "rag reply 1", "s2_gemma3_length": "3"},
        {"idx": "2", "language": "de", "subject": "b", "answer": "gold 2",
         "top_ce_score": "0.10", "n_ctx_used": "3",
         "s2_gemma3_prompt": "prompt with context 2",
         "s2_gemma3_response": "rag reply 2", "s2_gemma3_length": "3"},
    ])


@pytest.fixture
def no_context_frame():
    return pd.DataFrame([
        {"idx": "2", "language": "de", "answer": "gold 2",
         "s3_nc_prompt": "prompt without context 2",
         "s3_nc_response": "no-context reply 2"},
    ])


def test_rows_above_the_threshold_reuse_s2s_reply(s2_frame, no_context_frame):
    """The saving the plan asks for: S2 already generated that exact reply from
    that exact prompt."""
    out = reuse_s2_outputs_from(s2_frame, no_context_frame, 0.30)

    first = out[out["idx"] == "1"].iloc[0]
    assert first["route"] == "rag"
    assert first["s3_gemma3_crag_response"] == "rag reply 1"
    assert first["s3_gemma3_crag_prompt"] == "prompt with context 1"


def test_rows_below_the_threshold_take_the_no_context_reply(s2_frame,
                                                            no_context_frame):
    out = reuse_s2_outputs_from(s2_frame, no_context_frame, 0.30)

    second = out[out["idx"] == "2"].iloc[0]
    assert second["route"] == "no_context"
    assert second["s3_gemma3_crag_response"] == "no-context reply 2"
    assert second["s3_gemma3_crag_prompt"] == "prompt without context 2"


def test_the_reused_file_carries_the_label_columns_only(s2_frame,
                                                        no_context_frame):
    out = reuse_s2_outputs_from(s2_frame, no_context_frame, 0.30)

    assert "s2_gemma3_response" not in out.columns
    assert "s3_gemma3_crag_length" in out.columns
    assert list(out["s3_gemma3_crag_length"]) == [3, 3]
    # the row's own metadata survives
    assert list(out["answer"]) == ["gold 1", "gold 2"]
    assert list(out["language"]) == ["en", "de"]


def test_a_missing_no_context_generation_is_an_error(s2_frame):
    with pytest.raises(ValueError, match="no generation"):
        reuse_s2_outputs_from(s2_frame, pd.DataFrame([
            {"idx": "9", "s3_nc_response": "x", "s3_nc_prompt": "y"}]), 0.30)


def test_the_label_names_the_output_columns(s2_frame, no_context_frame):
    out = reuse_s2_outputs_from(s2_frame, no_context_frame, 0.30,
                                label="s3_custom")

    assert "s3_custom_response" in out.columns


def reuse_s2_outputs_from(s2, no_context, threshold, label="s3_gemma3_crag"):
    """Run reuse_s2_outputs on in-memory frames via temp CSVs."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        s2_path = Path(directory) / "s2.csv"
        nc_path = Path(directory) / "nc.csv"
        s2.to_csv(s2_path, index=False)
        no_context.to_csv(nc_path, index=False)
        return reuse_s2_outputs(str(s2_path), str(nc_path), threshold, label)


# ── the runner records the route ─────────────────────────────────────────────

class _FakeRetriever:
    def __init__(self, ce_score):
        self.ce_score = ce_score
        self.prompts = []

    def retrieve(self, subject, body, language, aspect_spans=None):
        return [{"subject": "Past", "response": "Answer",
                 "ce_score": self.ce_score}]

    def build_prompt(self, subject, body, retrieved, aspect_spans=None,
                     language="en"):
        prompt = f"ctx={len(retrieved)} {subject} Support reply:"
        self.prompts.append(prompt)
        return prompt


def _run(monkeypatch, tmp_path, ce_score, threshold):
    import src.generation.response_generator_v2 as generator

    retriever = _FakeRetriever(ce_score)
    monkeypatch.setattr(generator, "_make_retriever", lambda config: retriever)
    monkeypatch.setattr(generator, "generate_batch",
                        lambda prompts, tok, model, config, seq2seq=False,
                        row_ids=None: ["reply"] * len(prompts))
    monkeypatch.setattr("src.generation.model_backend.count_tokens_fn",
                        lambda tok: (lambda text: len(text.split())))
    df = pd.DataFrame([{"idx": 0, "language": "en", "subject": "s",
                        "body": "b", "answer": "gold", "prob_sub": "p",
                        "prob_statement": "p", "cause": "none",
                        "intent": "billing_issue", "ticket_type_nli": "Request",
                        "entities": "", "action_taken": "none",
                        "urgency": "low"}])
    config = generator.GenerationConfig(output_dir=str(tmp_path),
                                        max_input_tokens=4096,
                                        crag_threshold=threshold)
    return generator.run_rag_aspect(df, None, None, False, config,
                                    label="s3", crag=True), retriever


def test_the_runner_routes_a_weak_row_to_an_empty_context(monkeypatch, tmp_path):
    out, retriever = _run(monkeypatch, tmp_path, ce_score=0.05, threshold=0.30)

    assert out["route"].iloc[0] == "no_context"
    assert out["n_ctx_used"].iloc[0] == 0
    assert "ctx=0" in out["s3_prompt"].iloc[0]
    # the same template, just with no context
    assert out["s3_prompt"].iloc[0].endswith("Support reply:")


def test_the_runner_keeps_context_for_a_confident_row(monkeypatch, tmp_path):
    out, _ = _run(monkeypatch, tmp_path, ce_score=0.9, threshold=0.30)

    assert out["route"].iloc[0] == "rag"
    assert out["n_ctx_used"].iloc[0] == 1
    assert out["top_ce_score"].iloc[0] == 0.9


def test_without_crag_every_row_is_routed_through_rag(monkeypatch, tmp_path):
    import src.generation.response_generator_v2 as generator

    retriever = _FakeRetriever(0.01)
    monkeypatch.setattr(generator, "_make_retriever", lambda config: retriever)
    monkeypatch.setattr(generator, "generate_batch",
                        lambda prompts, tok, model, config, seq2seq=False,
                        row_ids=None: ["reply"] * len(prompts))
    monkeypatch.setattr("src.generation.model_backend.count_tokens_fn",
                        lambda tok: (lambda text: len(text.split())))
    df = pd.DataFrame([{"idx": 0, "language": "en", "subject": "s", "body": "b",
                        "answer": "gold"}])
    config = generator.GenerationConfig(output_dir=str(tmp_path),
                                        max_input_tokens=4096)

    out = generator.run_rag_aspect(df, None, None, False, config, label="s2")

    assert out["route"].iloc[0] == "rag"
    assert out["n_ctx_used"].iloc[0] == 1
