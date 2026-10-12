"""Step 4 (plan §5.6): labelled runners, the minimal-run modes, and the
Checkpointer writing into output_dir."""

import pandas as pd
import pytest

from src.generation.response_generator_v2 import (GenerationConfig,
                                                  _MINIMAL_RUN_MODES,
                                                  run_rag_aspect)
from src.utils import Checkpointer


# ── Checkpointer directory ───────────────────────────────────────────────────

def test_checkpointer_writes_into_the_given_directory(tmp_path):
    """On Kaggle only /kaggle/working survives, so the checkpoint has to live
    beside the run's output, not in the current directory."""
    ckpt = Checkpointer("s4_raft_aspect", every=2, dir=str(tmp_path / "gen"))

    assert ckpt.path == tmp_path / "gen" / ".ckpt_s4_raft_aspect.csv"
    assert ckpt.path.parent.is_dir()


def test_checkpointer_round_trips_rows(tmp_path):
    ckpt = Checkpointer("run", every=1, dir=str(tmp_path))
    ckpt.save([{"idx": "1", "generated_answer": "hello"}])

    assert ckpt.load() == [{"idx": "1", "generated_answer": "hello"}]

    ckpt.clear()
    assert ckpt.load() == []


def test_checkpointer_defaults_to_the_current_directory():
    assert Checkpointer("x").path.parent.name in ("", ".") or \
           str(Checkpointer("x").path) == ".ckpt_x.csv"


# ── the labelled runner ──────────────────────────────────────────────────────

class _FakeRetriever:
    """Returns two hits and records what it was asked for."""

    def __init__(self):
        self.calls = []

    def retrieve(self, subject, body, language, aspect_spans=None):
        self.calls.append({"language": language, "aspect_spans": aspect_spans})
        return [{"subject": "Past 1", "response": "Answer 1", "ce_score": 0.83},
                {"subject": "Past 2", "response": "Answer 2", "ce_score": 0.41}]

    def build_prompt(self, subject, body, retrieved, aspect_spans=None,
                     language="en"):
        aspects = "with-aspects" if aspect_spans else "no-aspects"
        return (f"{aspects} ctx={len(retrieved)} {subject} {body} "
                f"[{language}] Support reply:")


# a row that already carries every aspect column, like the minimal run's sample
_SAMPLE_ROW = {"idx": 0, "subject": "s", "body": "b", "language": "en",
               "answer": "gold", "prob_sub": "p", "prob_statement": "p",
               "cause": "none", "intent": "billing_issue",
               "ticket_type_nli": "Request", "entities": "",
               "action_taken": "none", "urgency": "low"}


@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {"idx": 0, "language": "en", "subject": "Login fails",
         "body": "I cannot log in", "answer": "Gold EN",
         "prob_sub": "Login fails", "cause": "none", "action_taken": "none",
         "intent": "login_or_access", "ticket_type_nli": "Incident",
         "entities": "", "urgency": "high", "prob_statement": "Login fails"},
        {"idx": 1, "language": "de", "subject": "Anmeldung schlägt fehl",
         "body": "Ich komme nicht rein", "answer": "Gold DE",
         "prob_sub": "Anmeldung schlägt fehl", "cause": "none",
         "action_taken": "none", "intent": "login_or_access",
         "ticket_type_nli": "Incident", "entities": "", "urgency": "low",
         "prob_statement": "Anmeldung schlägt fehl"},
    ])


def _patch_runner(monkeypatch, retriever):
    import src.generation.response_generator_v2 as generator

    monkeypatch.setattr(generator, "_make_retriever", lambda config: retriever)
    monkeypatch.setattr(generator, "generate_batch",
                        lambda prompts, tok, model, config, seq2seq=False,
                        row_ids=None: [f"reply {i}" for i in range(len(prompts))])
    monkeypatch.setattr("src.generation.model_backend.count_tokens_fn",
                        lambda tok: (lambda text: len(text.split())))


def test_label_names_the_output_columns(monkeypatch, sample_df, tmp_path):
    retriever = _FakeRetriever()
    _patch_runner(monkeypatch, retriever)
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)

    out = run_rag_aspect(sample_df, None, None, False, config,
                         label="s4_raft_aspect")

    assert list(out["idx"]) == [0, 1]
    for column in ("s4_raft_aspect_prompt", "s4_raft_aspect_response",
                   "s4_raft_aspect_length"):
        assert column in out.columns
    assert "rag_aspect_prompt" not in out.columns


def test_every_row_records_top_ce_score_and_n_ctx_used(monkeypatch, sample_df,
                                                       tmp_path):
    _patch_runner(monkeypatch, _FakeRetriever())
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)

    out = run_rag_aspect(sample_df, None, None, False, config, label="s2")

    assert list(out["top_ce_score"]) == [0.83, 0.83]
    assert list(out["n_ctx_used"]) == [2, 2]
    assert list(out["n_retrieved"]) == [2, 2]


def test_use_aspects_false_drops_the_aspects_from_retrieval_and_prompt(
        monkeypatch, sample_df, tmp_path):
    """S5: the RAFT adapter without aspects."""
    retriever = _FakeRetriever()
    _patch_runner(monkeypatch, retriever)
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)

    out = run_rag_aspect(sample_df, None, None, False, config, label="s5",
                         use_aspects=False)

    assert all(call["aspect_spans"] is None for call in retriever.calls)
    assert all("no-aspects" in prompt for prompt in out["s5_prompt"])


def test_use_aspects_true_passes_the_spans(monkeypatch, sample_df, tmp_path):
    retriever = _FakeRetriever()
    _patch_runner(monkeypatch, retriever)
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)

    out = run_rag_aspect(sample_df, None, None, False, config, label="s4")

    assert all(call["aspect_spans"] for call in retriever.calls)
    assert all("with-aspects" in prompt for prompt in out["s4_prompt"])


def test_the_runner_passes_each_row_language_to_the_prompt(monkeypatch,
                                                           sample_df, tmp_path):
    retriever = _FakeRetriever()
    _patch_runner(monkeypatch, retriever)
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)

    out = run_rag_aspect(sample_df, None, None, False, config, label="s2")

    assert "[en]" in out["s2_prompt"].iloc[0]
    assert "[de]" in out["s2_prompt"].iloc[1]
    assert [call["language"] for call in retriever.calls] == ["en", "de"]


def test_the_prompt_budget_drops_context_in_the_runner(monkeypatch, sample_df,
                                                       tmp_path):
    _patch_runner(monkeypatch, _FakeRetriever())
    # 17 "tokens" of budget leaves room for the prompt but not for both docs
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=17)

    out = run_rag_aspect(sample_df, None, None, False, config, label="s2")

    assert set(out["n_ctx_used"]) <= {0, 1}
    assert all(prompt.endswith("Support reply:") for prompt in out["s2_prompt"])


def test_the_runner_checkpoints_into_output_dir(monkeypatch, sample_df, tmp_path):
    _patch_runner(monkeypatch, _FakeRetriever())
    config = GenerationConfig(output_dir=str(tmp_path / "gen"),
                              generation_batch_size=1, checkpoint_every=1,
                              max_input_tokens=4096)

    run_rag_aspect(sample_df, None, None, False, config, label="s2")

    # cleared on a clean finish, but the directory was created where asked
    assert (tmp_path / "gen").is_dir()
    assert not (tmp_path / "gen" / ".ckpt_s2.csv").exists()


def test_a_resumed_run_skips_the_rows_already_done(monkeypatch, sample_df,
                                                   tmp_path):
    _patch_runner(monkeypatch, _FakeRetriever())
    config = GenerationConfig(output_dir=str(tmp_path), generation_batch_size=2,
                              max_input_tokens=4096)
    Checkpointer("s2", dir=str(tmp_path)).save(
        [{"idx": 0, "language": "en", "subject": "Login fails",
          "answer": "Gold EN", "generated_answer": "earlier reply"}])

    out = run_rag_aspect(sample_df, None, None, False, config, label="s2")

    assert len(out) == 2
    assert out.iloc[0]["generated_answer"] == "earlier reply"


def test_crag_is_not_available_before_step_9(monkeypatch, sample_df, tmp_path):
    _patch_runner(monkeypatch, _FakeRetriever())
    config = GenerationConfig(output_dir=str(tmp_path))

    with pytest.raises(NotImplementedError, match="CRAG"):
        run_rag_aspect(sample_df, None, None, False, config, crag=True)


# ── modes ────────────────────────────────────────────────────────────────────

def test_the_minimal_run_modes_cover_s1_to_s5():
    assert set(_MINIMAL_RUN_MODES) == {"rag_aspect", "rag_aspect_crag",
                                       "raft_aspect", "raft_rag"}
    assert _MINIMAL_RUN_MODES["raft_aspect"]["needs_adapter"]
    assert _MINIMAL_RUN_MODES["raft_rag"]["needs_adapter"]
    assert not _MINIMAL_RUN_MODES["raft_rag"]["use_aspects"]
    assert not _MINIMAL_RUN_MODES["rag_aspect"]["needs_adapter"]
    assert _MINIMAL_RUN_MODES["rag_aspect_crag"]["crag"]


def test_a_raft_mode_without_an_adapter_is_refused(tmp_path):
    from src.generation.response_generator_v2 import run

    sample = tmp_path / "sample.csv"
    pd.DataFrame([{"idx": 0, "subject": "s", "body": "b", "language": "en"}]
                 ).to_csv(sample, index=False)
    config = GenerationConfig(output_dir=str(tmp_path), adapter_path="")

    with pytest.raises(ValueError, match="needs a RAFT adapter"):
        run(str(sample), "raft_aspect", config)


def test_a_non_raft_mode_ignores_a_configured_adapter(monkeypatch, tmp_path):
    """S1-S3 must score the base model even if an adapter is left in the YAML."""
    import src.generation.response_generator_v2 as generator

    sample = tmp_path / "sample.csv"
    pd.DataFrame([_SAMPLE_ROW]).to_csv(sample, index=False)
    config = GenerationConfig(output_dir=str(tmp_path),
                              adapter_path="ranjankn/aspectforge-raft-gemma3",
                              skip_metrics=True)

    seen = {}

    def fake_load_model(cfg):
        seen["adapter_path"] = cfg.adapter_path
        return None, None, False

    monkeypatch.setattr(generator, "load_model", fake_load_model)
    monkeypatch.setattr(generator, "run_rag_aspect",
                        lambda *a, **k: pd.DataFrame([{"idx": 0}]))

    generator.run(str(sample), "rag_aspect", config, label="s2_gemma3")

    assert seen["adapter_path"] == ""
    assert (tmp_path / "s2_gemma3.csv").is_file()


def test_the_output_file_is_named_after_the_label(monkeypatch, tmp_path):
    import src.generation.response_generator_v2 as generator

    sample = tmp_path / "sample.csv"
    pd.DataFrame([_SAMPLE_ROW]).to_csv(sample, index=False)
    config = GenerationConfig(output_dir=str(tmp_path), skip_metrics=True)

    monkeypatch.setattr(generator, "load_model", lambda cfg: (None, None, False))
    monkeypatch.setattr(generator, "run_rag_aspect",
                        lambda *a, **k: pd.DataFrame([{"idx": 0}]))

    generator.run(str(sample), "rag_aspect", config, label="s1_gemma2")

    assert (tmp_path / "s1_gemma2.csv").is_file()


def test_a_sample_with_aspect_columns_is_not_re_extracted(monkeypatch, tmp_path):
    """plan §5.6: the minimal run's sample already carries the aspects, so no
    extractor runs and no *_with_aspects*.csv is written."""
    import src.generation.response_generator_v2 as generator

    sample = tmp_path / "sample_500.csv"
    pd.DataFrame([{"idx": 0, "subject": "s", "body": "b", "language": "en",
                   "answer": "gold", "prob_sub": "p", "prob_statement": "p",
                   "cause": "none", "intent": "billing_issue",
                   "ticket_type_nli": "Request", "entities": "",
                   "action_taken": "none", "urgency": "low"}]
                 ).to_csv(sample, index=False)
    config = GenerationConfig(output_dir=str(tmp_path), skip_metrics=True)

    def fail(*args, **kwargs):
        raise AssertionError("the aspect extractor must not run here")

    monkeypatch.setattr(generator, "enrich_dataframe_with_aspects", fail)
    monkeypatch.setattr(generator, "load_model", lambda cfg: (None, None, False))
    monkeypatch.setattr(generator, "run_rag_aspect",
                        lambda *a, **k: pd.DataFrame([{"idx": 0}]))

    generator.run(str(sample), "rag_aspect", config, label="s2")

    assert not list(tmp_path.glob("*_with_aspects*.csv"))


def test_no_aspects_flag_turns_the_aspect_path_off(monkeypatch, tmp_path):
    """S6 = rag_aspect --no_aspects."""
    import src.generation.response_generator_v2 as generator

    sample = tmp_path / "sample.csv"
    pd.DataFrame([_SAMPLE_ROW]).to_csv(sample, index=False)
    config = GenerationConfig(output_dir=str(tmp_path), skip_metrics=True)
    seen = {}

    monkeypatch.setattr(generator, "load_model", lambda cfg: (None, None, False))
    monkeypatch.setattr(generator, "run_rag_aspect",
                        lambda *a, **k: (seen.update(k) or pd.DataFrame([{"idx": 0}])))

    generator.run(str(sample), "rag_aspect", config, label="s6", no_aspects=True)

    assert seen["use_aspects"] is False
    assert seen["label"] == "s6"
