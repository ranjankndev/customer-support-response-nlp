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


# ── step 11: the two real jobs (plan §9, §K.2) ───────────────────────────────

import json

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("job,kernel_id", [
    ("prep_train", "ranjankumarnayak/aspectforge-prep-train"),
    ("generate_eval", "ranjankumarnayak/aspectforge-generate-eval"),
])
def test_the_pipeline_jobs_are_valid_t4x2_script_kernels(job, kernel_id):
    metadata, folder, _ = kaggle_job.read_job(job)

    assert metadata["id"] == kernel_id
    assert metadata["kernel_type"] == "script"
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["enable_gpu"] is True
    assert metadata["enable_internet"] is True
    assert metadata["is_private"] is True
    # the raw data comes straight from Kaggle, nothing is uploaded
    assert metadata["dataset_sources"] == [
        "tobiasbueck/multilingual-customer-support-tickets"]
    assert (folder / "run.py").is_file()


def test_job_two_reads_job_ones_output():
    """plan §K.2: the adapter and the index are job 1's kernel output."""
    metadata, _, _ = kaggle_job.read_job("generate_eval")

    assert metadata["kernel_sources"] == [
        "ranjankumarnayak/aspectforge-prep-train"]
    assert kaggle_job.read_job("prep_train")[0]["kernel_sources"] == []


@pytest.mark.parametrize("job", ["prep_train", "generate_eval"])
def test_the_job_scripts_bake_in_the_exact_commit(job):
    _, folder, metadata_path = kaggle_job.read_job(job)
    staged = kaggle_job.stage_job(folder, metadata_path, "a" * 40,
                                  "v3-minimal-run")
    try:
        source = (Path(staged.name) / "run.py").read_text(encoding="utf-8")

        assert f'COMMIT = "{"a" * 40}"' in source
        assert 'BRANCH = "v3-minimal-run"' in source
        assert "__ASPECTFORGE_" not in source
        # the script checks out that sha and refuses to run on anything else
        assert "Requested commit" in source
    finally:
        staged.cleanup()


@pytest.mark.parametrize("job", ["prep_train", "generate_eval"])
def test_the_job_scripts_are_syntactically_valid(job):
    import ast

    _, folder, metadata_path = kaggle_job.read_job(job)
    staged = kaggle_job.stage_job(folder, metadata_path, "b" * 40, "main")
    try:
        source = (Path(staged.name) / "run.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name for node in ast.walk(tree)
                     if isinstance(node, ast.FunctionDef)}

        # every job writes STATUS.json after each step, so `pull` can diagnose
        assert {"update_status", "log", "shell", "main"} <= functions
    finally:
        staged.cleanup()


@pytest.mark.parametrize("job", ["prep_train", "generate_eval"])
def test_no_job_hard_codes_a_secret(job):
    """plan §K.2: no secrets on Kaggle."""
    source = (ROOT / "kaggle" / job / "run.py").read_text(encoding="utf-8")

    assert "hf_" not in source.lower().replace("hf_token", "")
    for forbidden in ("HF_TOKEN=", "api_key", "secret_key"):
        assert forbidden not in source


def test_job_one_does_not_push_to_hugging_face():
    """The adapter is pushed locally after `pull`, where the token lives."""
    source = (ROOT / "kaggle" / "prep_train" / "run.py").read_text(
        encoding="utf-8")

    assert "--no_push" in source
    assert "push_to_hub" not in source


def test_job_one_covers_every_preparation_step():
    source = (ROOT / "kaggle" / "prep_train" / "run.py").read_text(
        encoding="utf-8")

    for step in ("src.preprocessing", "src.minimal_run", "aspect_extractor",
                 "src.retrieval.indexer", "src.aspect.aspect_heads",
                 "src.retrieval.eval_retrieval", "src.step3_raft"):
        assert step in source, f"job 1 never runs {step}"


def test_job_two_generates_every_system_and_reports():
    source = (ROOT / "kaggle" / "generate_eval" / "run.py").read_text(
        encoding="utf-8")

    for label in ("s1_gemma2", "s2_gemma3", "s3_gemma3_crag",
                  "s4_raft_aspect", "s5_raft_noaspect",
                  "s6_gemma3_noaspect"):
        assert label in source, f"job 2 never produces {label}"
    assert "crag_tune" in source and "crag_reuse" in source
    assert '"report"' in source


def test_job_two_uses_both_gpus():
    source = (ROOT / "kaggle" / "generate_eval" / "run.py").read_text(
        encoding="utf-8")

    assert '"CUDA_VISIBLE_DEVICES": "0"' in source
    assert '"CUDA_VISIBLE_DEVICES": "1"' in source


def test_the_kaggle_requirements_leave_the_unsloth_stack_alone():
    """plan §9: listing torch/transformers/trl/peft here breaks Unsloth's T4
    fp16 path, because pip would resolve them a second time."""
    text = (ROOT / "requirements-kaggle.txt").read_text(encoding="utf-8")
    packages = {line.split("#")[0].strip().lower()
                for line in text.splitlines()
                if line.strip() and not line.strip().startswith("#")}

    assert {"sentence-transformers", "faiss-cpu", "bm25s", "rouge-score",
            "sacrebleu", "bert-score", "langdetect", "pyyaml", "pandas",
            "matplotlib"} <= packages
    for managed in ("torch", "transformers", "trl", "peft", "unsloth"):
        assert managed not in packages
