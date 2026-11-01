"""Step 6 (plan §6.1): the fixed samples and attach_aspects."""

from pathlib import Path

import pandas as pd
import pytest

from src.minimal_run import (ASPECT_COLUMNS, MinimalRunConfig, SampleSpec,
                             attach_aspects, build_sample, cmd_sample,
                             file_hash, stratified_sample)


def _split(n_en: int, n_de: int, start: int = 0) -> pd.DataFrame:
    rows = []
    idx = start
    for language, count in (("en", n_en), ("de", n_de)):
        for i in range(count):
            rows.append({"idx": str(idx), "language": language,
                         "subject": f"{language} subject {i}",
                         "body": f"{language} body {i}",
                         "answer": f"{language} answer {i}",
                         "queue": "Billing", "type": "Request",
                         "priority": "low"})
            idx += 1
    return pd.DataFrame(rows)


# ── stratified_sample ────────────────────────────────────────────────────────

def test_a_500_row_sample_is_250_en_plus_250_de():
    """The plan's acceptance test."""
    sample = stratified_sample(_split(3000, 3000), 500, "language", 42)

    assert len(sample) == 500
    assert sample["language"].value_counts().to_dict() == {"en": 250, "de": 250}


def test_sampling_is_reproducible_for_a_seed():
    population = _split(1000, 1000)

    first = stratified_sample(population, 100, "language", 42)
    second = stratified_sample(population, 100, "language", 42)
    other_seed = stratified_sample(population, 100, "language", 7)

    assert list(first["idx"]) == list(second["idx"])
    assert list(first["idx"]) != list(other_seed["idx"])


def test_the_sample_is_ordered_by_idx():
    sample = stratified_sample(_split(100, 100), 20, "language", 42)

    numeric = [int(i) for i in sample["idx"]]
    assert numeric == sorted(numeric)


def test_an_odd_n_spreads_the_remainder():
    sample = stratified_sample(_split(50, 50), 7, "language", 42)

    assert len(sample) == 7
    assert sorted(sample["language"].value_counts().tolist()) == [3, 4]


def test_a_small_group_contributes_all_its_rows_and_the_total_still_holds():
    sample = stratified_sample(_split(100, 10), 50, "language", 42)

    counts = sample["language"].value_counts().to_dict()
    assert len(sample) == 50
    assert counts["de"] == 10
    assert counts["en"] == 40


def test_asking_for_more_rows_than_exist_returns_everything():
    population = _split(5, 5)

    assert len(stratified_sample(population, 100, "language", 42)) == 10
    assert stratified_sample(population, 0, "language", 42).empty


def test_sampling_without_the_stratify_column_still_works():
    population = _split(20, 20).drop(columns=["language"])

    assert len(stratified_sample(population, 10, "language", 42)) == 10


# ── build_sample reuse ───────────────────────────────────────────────────────

def test_an_existing_sample_is_reused_with_the_same_hash(tmp_path, capsys):
    """The plan's acceptance test: a rerun must not resample."""
    source = tmp_path / "test.csv"
    _split(100, 100).to_csv(source, index=False)
    spec = SampleSpec("test_500", str(source), 20, str(tmp_path / "test_20.csv"))

    first = build_sample(spec, "language", 42)
    first_hash = file_hash(spec.out_path)
    second = build_sample(spec, "language", 42)

    assert list(first["idx"]) == list(second["idx"])
    assert file_hash(spec.out_path) == first_hash
    assert "reusing" in capsys.readouterr().out


def test_force_resamples_with_a_fresh_file(tmp_path):
    source = tmp_path / "test.csv"
    _split(100, 100).to_csv(source, index=False)
    spec = SampleSpec("test_500", str(source), 20, str(tmp_path / "s.csv"))

    build_sample(spec, "language", 42)
    before = file_hash(spec.out_path)
    build_sample(spec, "language", 7, force=True)

    assert file_hash(spec.out_path) != before


def test_a_missing_split_says_how_to_build_it(tmp_path):
    spec = SampleSpec("test_500", str(tmp_path / "nope.csv"), 10,
                      str(tmp_path / "out.csv"))

    with pytest.raises(FileNotFoundError, match="preprocessing"):
        build_sample(spec, "language", 42)


def test_idx_is_added_when_a_split_has_none(tmp_path):
    source = tmp_path / "test.csv"
    _split(10, 10).drop(columns=["idx"]).to_csv(source, index=False)
    spec = SampleSpec("test_500", str(source), 6, str(tmp_path / "out.csv"))

    sample = build_sample(spec, "language", 42)

    assert "idx" in sample.columns
    assert sample["idx"].nunique() == 6


# ── cmd_sample ───────────────────────────────────────────────────────────────

def test_cmd_sample_writes_all_three_samples(tmp_path):
    for name, frame in (("train", _split(2000, 2000, start=10000)),
                        ("validate", _split(600, 600, start=5000)),
                        ("test", _split(600, 600, start=0))):
        frame.to_csv(tmp_path / f"cs-support-multi-{name}.csv", index=False)
    config = MinimalRunConfig(
        sample_dir=str(tmp_path / "minimal"),
        train_csv=str(tmp_path / "cs-support-multi-train.csv"),
        validate_csv=str(tmp_path / "cs-support-multi-validate.csv"),
        test_csv=str(tmp_path / "cs-support-multi-test.csv"),
        test_n=500, val_n=500, raft_n=300)

    written = cmd_sample(config)

    assert set(written) == {"test_500", "val_500", "raft_train_300"}
    test = pd.read_csv(written["test_500"], dtype=str)
    assert len(test) == 500
    assert test["language"].value_counts().to_dict() == {"en": 250, "de": 250}
    assert len(pd.read_csv(written["raft_train_300"], dtype=str)) == 300


def test_cmd_sample_refuses_overlapping_samples(tmp_path):
    """A RAFT row that is also an evaluation row would leak (plan §7.4)."""
    shared = _split(400, 400)
    for name in ("train", "validate", "test"):
        shared.to_csv(tmp_path / f"cs-support-multi-{name}.csv", index=False)
    config = MinimalRunConfig(
        sample_dir=str(tmp_path / "minimal"),
        train_csv=str(tmp_path / "cs-support-multi-train.csv"),
        validate_csv=str(tmp_path / "cs-support-multi-validate.csv"),
        test_csv=str(tmp_path / "cs-support-multi-test.csv"),
        test_n=100, val_n=100, raft_n=200)

    with pytest.raises(AssertionError, match="overlap|share"):
        cmd_sample(config)


# ── attach_aspects ───────────────────────────────────────────────────────────

def _aspects_frame(ids):
    return pd.DataFrame([
        {"idx": str(i), "language": "en", "subject": "s", "body": "b",
         "prob_sub": f"problem {i}", "prob_statement": f"statement {i}",
         "cause": "none", "intent": "billing_issue",
         "ticket_type_nli": "Request", "entities": "", "action_taken": "none",
         "urgency": "low"} for i in ids])


def test_attach_aspects_merges_by_idx_and_keeps_the_gold_answer():
    """The extractor drops `answer`, so the sample is the left side."""
    sample = _split(2, 2)
    merged = attach_aspects(sample, _aspects_frame([0, 1, 2, 3]))

    assert len(merged) == 4
    assert list(merged["answer"]) == list(sample["answer"])
    for column in ASPECT_COLUMNS:
        assert column in merged.columns
    assert merged.loc[merged["idx"] == "2", "prob_sub"].iloc[0] == "problem 2"


def test_attach_aspects_fills_missing_rows_with_none():
    merged = attach_aspects(_split(2, 2), _aspects_frame([0, 1]))

    assert list(merged["prob_sub"]) == ["problem 0", "problem 1", "none", "none"]


def test_attach_aspects_replaces_stale_aspect_columns():
    sample = _split(1, 1)
    sample["prob_sub"] = "stale"

    merged = attach_aspects(sample, _aspects_frame([0, 1]))

    assert "prob_sub_x" not in merged.columns
    assert list(merged["prob_sub"]) == ["problem 0", "problem 1"]


def test_attach_aspects_rejects_duplicate_idx():
    aspects = pd.concat([_aspects_frame([0]), _aspects_frame([0])])

    with pytest.raises(ValueError, match="duplicate idx"):
        attach_aspects(_split(1, 0), aspects)


def test_attach_aspects_needs_an_idx_column():
    with pytest.raises(ValueError, match="idx"):
        attach_aspects(_split(1, 0).drop(columns=["idx"]), _aspects_frame([0]))


def test_attach_aspects_needs_at_least_one_aspect_column():
    aspects = pd.DataFrame([{"idx": "0", "subject": "s"}])

    with pytest.raises(ValueError, match="aspect columns"):
        attach_aspects(_split(1, 0), aspects)


def test_attach_aspects_writes_a_file_the_generator_can_use(tmp_path):
    from src.minimal_run import cmd_attach_aspects

    sample_path = tmp_path / "test_500.csv"
    aspects_path = tmp_path / "test_500_aspects.csv"
    _split(2, 2).to_csv(sample_path, index=False)
    _aspects_frame([0, 1, 2, 3]).to_csv(aspects_path, index=False)

    out = cmd_attach_aspects(str(sample_path), str(aspects_path))

    assert Path(out).name == "test_500_with_aspects.csv"
    written = pd.read_csv(out, dtype=str).fillna("")
    assert set(["idx", "language", "subject", "body", "answer"]
               + ASPECT_COLUMNS) <= set(written.columns)


# ── config ───────────────────────────────────────────────────────────────────

def test_the_kaggle_overrides_redirect_the_sample_paths():
    from src.minimal_run import load_minimal_config

    config = load_minimal_config("config/config_preprocess.yaml",
                                 "config/kaggle.yaml")

    assert config.train_csv.startswith("/kaggle/input/aspectforge-data")
    assert config.sample_dir.startswith("/kaggle/working/Code")
    assert config.output_dir.startswith("/kaggle/working/Code")
    assert config.seed == 42


def test_the_local_config_keeps_relative_paths():
    from src.minimal_run import load_minimal_config

    config = load_minimal_config("config/config_preprocess.yaml")

    assert config.train_csv == "data/processed/cs-support-multi-train.csv"
    assert config.sample_dir == "data/minimal"
