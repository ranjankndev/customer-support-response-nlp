"""kaggle/prep_train/run.py: reuse of an earlier run's aspects + index."""
import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def job(tmp_path, monkeypatch):
    saved = dict(os.environ)
    spec = importlib.util.spec_from_file_location(
        "prep_train_run", ROOT / "kaggle" / "prep_train" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    os.environ.clear()
    os.environ.update(saved)            # the import sets GPU env defaults
    monkeypatch.setattr(module, "INPUT_ROOT", tmp_path / "input")
    monkeypatch.setattr(module, "CODE", tmp_path / "Code")
    monkeypatch.setattr(module, "log", lambda message: None)
    minimal = tmp_path / "Code" / "data" / "minimal"
    minimal.mkdir(parents=True)
    cache = tmp_path / "input" / module.CACHE_SLUG
    (cache / "index" / "bm25").mkdir(parents=True)
    for name in module.CACHED_SAMPLES:
        (minimal / f"{name}.csv").write_text(f"idx\n{name}\n")
        (cache / f"{name}.csv").write_text(f"idx\n{name}\n")
        (cache / f"{name}_aspects.csv").write_text("aspects")
        (cache / f"{name}_with_aspects.csv").write_text("with")
    for name in module.CACHED_INDEX_FILES:
        (cache / "index" / name).write_bytes(b"f")
    (cache / "index" / "bm25" / "params.index.json").write_text("{}")
    (cache / "aspect_table.csv").write_text("label")
    return module, minimal


def test_cache_is_used_when_samples_match(job):
    module, minimal = job
    assert module.restore_cached_aspects()
    assert (minimal / "val_500_with_aspects.csv").read_text() == "with"
    assert module.restore_cached_index()
    index = module.CODE / "data" / "index"
    assert (index / "faiss_index.bin").is_file()
    assert (index / "bm25" / "params.index.json").is_file()
    assert (module.CODE / "output" / "report" / "aspect_table.csv").is_file()


def test_cache_is_ignored_when_a_sample_differs(job):
    module, minimal = job
    (minimal / "test_500.csv").write_text("idx\nsomething else\n")
    assert not module.restore_cached_aspects()
    assert not (minimal / "test_500_aspects.csv").exists()


def test_no_cache_attached(job, tmp_path):
    module, _ = job
    import shutil
    shutil.rmtree(tmp_path / "input")
    (tmp_path / "input").mkdir()
    assert not module.restore_cached_aspects()


def test_index_cache_incomplete_rebuilds(job, tmp_path):
    module, _ = job
    (tmp_path / "input" / module.CACHE_SLUG / "index" / "dense.npy").unlink()
    assert not module.restore_cached_index()


def test_raft_examples_restored_only_after_the_aspects(job, tmp_path):
    module, _ = job
    raft = tmp_path / "input" / module.RAFT_CACHE_SLUG
    raft.mkdir(parents=True)
    (raft / "raft_train.jsonl").write_text("{}")
    calls = []
    module.shell = lambda command, **kw: calls.append(command)
    target = module.CODE / "output" / "raft" / "raft_train.jsonl"

    module._RESTORED["aspects"] = False
    module.raft_build()
    assert calls and not target.exists()        # built from scratch

    calls.clear()
    module._RESTORED["aspects"] = True
    module.raft_build()
    assert not calls and target.read_text() == "{}"
