"""src/preprocessing.py: idx must stay unique across the three splits."""
import sys

import pandas as pd
import yaml

from src import preprocessing


def _run(tmp_path, monkeypatch, renumber):
    raw = pd.DataFrame({
        "subject": [f"s{i}" for i in range(60)],
        "body": [f"b{i}" for i in range(60)],
        "answer": [f"a{i}" for i in range(60)],
        "language": ["en", "de"] * 30,
    })
    raw.to_csv(tmp_path / "raw.csv", index=False)
    config = {"input_csv": str(tmp_path / "raw.csv"),
              "output_dir": str(tmp_path / "out"),
              "split": {"test_size": 10, "validate_size": 10,
                        "stratify_col": "language", "random_seed": 42,
                        "renumber_idx": renumber}}
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.setattr(sys, "argv",
                        ["preprocessing", "--config", str(tmp_path / "cfg.yaml")])
    preprocessing.main()
    return [pd.read_csv(tmp_path / "out" / f"cs-support-multi-{name}.csv",
                        dtype=str)
            for name in ("train", "validate", "test")]


def test_global_idx_is_unique_across_splits(tmp_path, monkeypatch):
    train, val, test = _run(tmp_path, monkeypatch, renumber=False)
    ids = pd.concat([train["idx"], val["idx"], test["idx"]])
    assert len(ids) == 60 and ids.is_unique
    # the idx still points at the same raw row
    assert (test["subject"] == "s" + test["idx"]).all()


def test_old_renumbering_is_reachable(tmp_path, monkeypatch):
    _, val, test = _run(tmp_path, monkeypatch, renumber=True)
    assert list(test["idx"]) == [str(i) for i in range(10)]
    assert set(val["idx"]) & set(test["idx"])
