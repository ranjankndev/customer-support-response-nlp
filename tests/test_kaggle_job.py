from pathlib import Path

import pytest

from scripts import kaggle_job


def test_hello_kernel_metadata_is_valid():
    metadata, folder, metadata_path = kaggle_job.read_job("hello")

    assert metadata["id"] == "ranjankumarnayak/aspectforge-hello"
    assert metadata["code_file"] == "run.py"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert (folder / "run.py").is_file()
    assert metadata_path.is_file()


def test_job_name_cannot_escape_kaggle_directory():
    with pytest.raises(RuntimeError, match="Job name"):
        kaggle_job.read_job("../outside")


def test_push_staging_bakes_exact_commit_and_branch():
    _, folder, metadata_path = kaggle_job.read_job("hello")
    staged = kaggle_job.stage_job(folder, metadata_path, "0123456789abcdef", "v3-minimal-run")
    try:
        source = (Path(staged.name) / "run.py").read_text(encoding="utf-8")

        assert "0123456789abcdef" in source
        assert 'BRANCH = "v3-minimal-run"' in source
        assert "__ASPECTFORGE_" not in source
        assert (Path(staged.name) / "kernel-metadata.json").is_file()
    finally:
        staged.cleanup()
