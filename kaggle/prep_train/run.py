"""
Kaggle job 1 — prep + RAFT train (plan §9 notebook 1, run as a script kernel).

Steps, each logged to /kaggle/working/logs/ and recorded in STATUS.json so a
failure can be diagnosed from `kaggle_job.py pull` alone:

  1 splits          regenerate the stratified splits from the raw Kaggle dataset
  2 sample          the three fixed samples (test_500, val_500, raft_train)
  3 aspects         aspect extraction for each sample (GPU 0 and GPU 1 in parallel)
  4 index           BGE-M3 hybrid index + dense.npy, then the aspect heads
  5 eval_retrieval  the four retrieval configurations on val_500
  6 raft_build      the RAFT JSONL
  7 raft_train      Unsloth QLoRA; the adapter is a kernel output

The adapter is NOT pushed to HF from here: no secrets live on Kaggle, so the
push happens locally after `kaggle_job.py pull` (plan §K.2).
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

REPO_URL = "__ASPECTFORGE_REPO_URL__"
BRANCH = "__ASPECTFORGE_BRANCH__"
COMMIT = "__ASPECTFORGE_COMMIT__"
JOB = "__ASPECTFORGE_JOB__"

WORK = Path("/kaggle/working")
CODE = WORK / "Code"
LOG_DIR = WORK / "logs"
STATUS_PATH = WORK / "STATUS.json"
INPUT_ROOT = Path("/kaggle/input")
RAW_SLUG = "aspectforge-cs-support"      # mount path varies by Kaggle version
# the aspect SFT adapter (HF ranjan56cse/cs-support-labels is private and no
# HF token lives on Kaggle), copied into a private dataset
ADAPTER_SLUG = "aspectforge-aspect-sft"
# optional: aspects + index from an earlier run of this job (a private dataset),
# reused only when the fresh samples are byte-identical to the cached ones
CACHE_SLUG = "aspectforge-prep-cache"
RAFT_CACHE_SLUG = "aspectforge-raft-cache"      # raft_train.jsonl of a run
CACHED_SAMPLES = ("test_500", "val_500", "raft_train_3000")

# The Kaggle image ships TensorFlow and JAX; either can grab most of a T4 the
# moment it initialises, which starved the reranker (attempt 5 OOM). Every
# child process inherits these.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_JAX", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
RAW_DIR = INPUT_ROOT
OVERRIDES = "config/kaggle.yaml"

_STATE: dict = {"job": JOB, "branch": BRANCH, "commit": COMMIT, "steps": []}


# ── status and logging ────────────────────────────────────────────────────────

def update_status(step: str, ok: bool, error: str | None = None) -> None:
    entry = {"step": step, "ok": ok, "error": error,
             "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    _STATE["steps"] = [s for s in _STATE["steps"] if s["step"] != step] + [entry]
    _STATE.update(step=step, ok=ok, error=error,
                  updated_at=entry["updated_at"])
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(_STATE, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / f"{JOB}.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def shell(command: list[str], *, step: str, cwd: Path = CODE,
          env: dict | None = None, check: bool = True) -> int:
    """Run one command, streaming its output into logs/<step>.log."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step}.log"
    log(f"$ {' '.join(command)}  (-> {log_path.name})")
    merged = {**os.environ, **(env or {})}
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(cwd), env=merged,
                                stdout=stream, stderr=subprocess.STDOUT,
                                text=True)
    if check and result.returncode:
        raise RuntimeError(f"{step}: command exited {result.returncode}: "
                           f"{' '.join(command)} (see logs/{step}.log)")
    return result.returncode


def run_step(name: str, function) -> None:
    update_status(name, False)
    log(f"=== {name} ===")
    function()
    update_status(name, True)
    log(f"=== {name} done ===")


def parallel(commands: list[tuple[str, list[str], dict]]) -> None:
    """
    Run commands at the same time, one per GPU. The T4x2 machine has two cards,
    and the aspect extractor is the long serial step, so the samples are split
    across CUDA_VISIBLE_DEVICES 0 and 1.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    processes = []
    for step, command, env in commands:
        log_path = LOG_DIR / f"{step}.log"
        log(f"$ (parallel) {' '.join(command)}  (-> {log_path.name})")
        stream = log_path.open("a", encoding="utf-8")
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        processes.append((step, subprocess.Popen(
            command, cwd=str(CODE), env={**os.environ, **env},
            stdout=stream, stderr=subprocess.STDOUT, text=True), stream))
    failures = []
    for step, process, stream in processes:
        code = process.wait()
        stream.close()
        if code:
            failures.append(f"{step} exited {code} (see logs/{step}.log)")
    if failures:
        raise RuntimeError("; ".join(failures))


# ── steps ─────────────────────────────────────────────────────────────────────

def checkout() -> None:
    CODE.parent.mkdir(parents=True, exist_ok=True)
    if not CODE.exists():
        shell(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL,
               str(CODE)], step="checkout", cwd=WORK)
    shell(["git", "fetch", "--depth", "1", "origin", COMMIT], step="checkout")
    shell(["git", "checkout", "--detach", COMMIT], step="checkout")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(CODE),
                            check=True, text=True,
                            stdout=subprocess.PIPE).stdout.strip()
    if actual != COMMIT:
        raise RuntimeError(f"Requested commit {COMMIT}, checked out {actual}")
    log(f"Code at {COMMIT}")


def install() -> None:
    # Unsloth first, so it chooses the torch/transformers/trl/peft versions.
    shell([sys.executable, "-m", "pip", "install", "-q", "unsloth"],
          step="install")
    shell([sys.executable, "-m", "pip", "install", "-q", "-r",
           "requirements-kaggle.txt"], step="install")


def gpu_check() -> None:
    shell(["nvidia-smi"], step="gpu_check")


def smoke() -> None:
    """Plan §10 step 3 acceptance: bos / German / budget on the real GPU models."""
    shell([sys.executable, "kaggle/prep_train/smoke.py"], step="smoke")
    shutil.copy(CODE / "smoke_results.json", WORK / "smoke_results.json")


def find_raw_csv() -> Path:
    """
    The raw dataset ships several CSVs; the one that reproduces the report's
    28,237 EN+DE rows is the widest multilingual file. Pick the candidate with
    the most EN+DE rows and record the choice in the log.
    """
    import pandas as pd

    everything = sorted(INPUT_ROOT.rglob("*.csv"))
    log(f"CSV files under {INPUT_ROOT}: {[str(p) for p in everything][:20]}")
    candidates = [p for p in everything if RAW_SLUG in str(p)] or everything
    if not candidates:
        raise RuntimeError(f"No CSV under {RAW_DIR}")
    best, best_rows = None, -1
    for path in candidates:
        try:
            frame = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception as exc:        # a malformed extra file is not fatal
            log(f"  {path.name}: unreadable ({exc})")
            continue
        language_column = next((c for c in frame.columns
                                if c.lower() == "language"), None)
        if language_column is None or "subject" not in frame.columns:
            log(f"  {path.name}: {len(frame)} rows, no language/subject column")
            continue
        rows = int(frame[language_column].str.lower().str[:2]
                   .isin(["en", "de"]).sum())
        log(f"  {path.name}: {len(frame)} rows, {rows} EN+DE")
        if rows > best_rows:
            best, best_rows = path, rows
    if best is None:
        raise RuntimeError(f"No usable CSV under {RAW_DIR}")
    log(f"Raw CSV chosen: {best} ({best_rows} EN+DE rows)")
    return best


def find_adapter_dir() -> Path:
    """The folder holding adapter_config.json in the attached adapter dataset."""
    configs = sorted(INPUT_ROOT.rglob("adapter_config.json"))
    preferred = [p for p in configs if ADAPTER_SLUG in str(p)] or configs
    if not preferred:
        raise RuntimeError(f"No adapter_config.json under {INPUT_ROOT}; attach "
                           f"the {ADAPTER_SLUG} dataset")
    log(f"Aspect adapter: {preferred[0].parent}")
    return preferred[0].parent


def _cache_file(name: str, slug: str = CACHE_SLUG) -> Path | None:
    hits = [p for p in INPUT_ROOT.rglob(name) if slug in str(p)]
    return hits[0] if hits else None


def restore_cached_aspects() -> bool:
    """Copy cached *_aspects / *_with_aspects CSVs if the samples match."""
    minimal = CODE / "data" / "minimal"
    pairs = []
    for name in CACHED_SAMPLES:
        cached = _cache_file(f"{name}.csv")
        fresh = minimal / f"{name}.csv"
        if cached is None:
            return False
        if not fresh.is_file() or cached.read_bytes() != fresh.read_bytes():
            log(f"Cache ignored: {name}.csv differs from the cached sample")
            return False
        for suffix in ("_aspects.csv", "_with_aspects.csv"):
            source = _cache_file(f"{name}{suffix}")
            if source is None:
                log(f"Cache ignored: no {name}{suffix}")
                return False
            pairs.append((source, minimal / f"{name}{suffix}"))
    for source, target in pairs:
        shutil.copy(source, target)
    log(f"Aspects restored from {CACHE_SLUG} ({len(pairs)} files)")
    return True


CACHED_INDEX_FILES = ("faiss_index.bin", "metadata.json", "dense.npy",
                      "aspect_heads.joblib")


def restore_cached_index() -> bool:
    """Files are found by name: Kaggle may flatten the dataset's folders."""
    files = {name: _cache_file(name) for name in CACHED_INDEX_FILES}
    bm25_params = _cache_file("params.index.json")
    table = _cache_file("aspect_table.csv")
    missing = [n for n, p in files.items() if p is None]
    if missing or bm25_params is None or table is None:
        log(f"Index cache incomplete (missing {missing or 'bm25/table'}); rebuilding")
        return False
    index_dir = CODE / "data" / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True)
    for name, source in files.items():
        shutil.copy(source, index_dir / name)
    shutil.copytree(bm25_params.parent, index_dir / "bm25")
    report = CODE / "output" / "report"
    report.mkdir(parents=True, exist_ok=True)
    shutil.copy(table, report / "aspect_table.csv")
    log(f"Index + aspect heads restored from {CACHE_SLUG}")
    return True


_RESTORED = {"aspects": False}


def splits() -> None:
    raw = find_raw_csv()
    data_dir = CODE / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / "cs-support-multi.csv"
    target.write_bytes(raw.read_bytes())
    shell([sys.executable, "-m", "src.preprocessing", "--config",
           "config/config_preprocess.yaml"], step="splits")


def sample() -> None:
    shell([sys.executable, "-m", "src.minimal_run", "sample",
           "--config", "config/config_preprocess.yaml",
           "--sample_dir", "data/minimal"], step="sample")


def aspects() -> None:
    """Aspect extraction per sample, two GPUs at a time."""
    jobs = []
    for name, device in (("test_500", "0"), ("val_500", "1"),
                         ("raft_train_3000", "0")):
        source = CODE / "data" / "minimal" / f"{name}.csv"
        if not source.is_file():
            log(f"  {name}: no sample file, skipped")
            continue
        jobs.append((name, device, source))

    if restore_cached_aspects():
        _RESTORED["aspects"] = True
        return
    adapter = find_adapter_dir()
    # the two 500-row samples run together, the big one on its own afterwards
    for batch in (jobs[:2], jobs[2:]):
        commands = []
        for name, device, source in batch:
            out = f"data/minimal/{name}_aspects.csv"
            commands.append((
                f"aspects_{name}",
                [sys.executable, "-m", "src.aspect.aspect_extractor",
                 "--input", str(source.relative_to(CODE)), "--out", out,
                 "--labels", "sft",
                 "--adapter", str(adapter), "--adapter_subfolder", ""],
                {"CUDA_VISIBLE_DEVICES": device}))
        if commands:
            parallel(commands)

    for name, _, _ in jobs:
        shell([sys.executable, "-m", "src.minimal_run", "attach_aspects",
               "--sample", f"data/minimal/{name}.csv",
               "--aspects", f"data/minimal/{name}_aspects.csv",
               "--out", f"data/minimal/{name}_with_aspects.csv"],
              step=f"attach_{name}")


def index() -> None:
    # same samples => same seed-42 splits, so the cached index still matches
    if _RESTORED["aspects"] and restore_cached_index():
        return
    # The train aspects are the index's own; the evaluation samples must not be
    # in it, and the indexer asserts that.
    shell([sys.executable, "-m", "src.retrieval.indexer",
           "--config", "config/retrieval_config.yaml"], step="index")
    shell([sys.executable, "-m", "src.aspect.aspect_heads", "train",
           "--config", "config/retrieval_config.yaml"], step="aspect_heads")
    shell([sys.executable, "-m", "src.aspect.aspect_heads", "eval",
           "--config", "config/retrieval_config.yaml",
           "--eval_csv", "data/minimal/val_500_with_aspects.csv",
           "--sft_aspects_csv", "data/minimal/val_500_aspects.csv"],
          step="aspect_heads")


def eval_retrieval() -> None:
    shell([sys.executable, "-m", "src.retrieval.eval_retrieval",
           "--config", "config/inference_retrieval_config.yaml",
           "--input", "data/minimal/val_500_with_aspects.csv",
           "--overrides", OVERRIDES,
           "--label", "m3_hybrid_rerank"], step="eval_retrieval")


def raft_build() -> None:
    # the build reads only the samples and the index, so it is reusable
    # exactly when those were restored
    cached = (_cache_file("raft_train.jsonl", RAFT_CACHE_SLUG)
              if _RESTORED["aspects"] else None)
    if cached is not None:
        target = CODE / "output" / "raft" / "raft_train.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(cached, target)
        log(f"RAFT examples restored from {RAFT_CACHE_SLUG}")
        return
    shell([sys.executable, "-m", "src.step3_raft", "build",
           "--config", "config/config_raft.yaml"], step="raft_build")


def raft_train() -> None:
    # --no_push: the HF token stays on the local machine (plan §K.2)
    shell([sys.executable, "-m", "src.step3_raft", "train",
           "--config", "config/config_raft.yaml", "--no_push"],
          step="raft_train")


def collect_outputs() -> None:
    """Copy what job 2 and the local machine need out of Code/."""
    for relative in ("data/minimal", "data/index", "output/raft", "output/report"):
        source = CODE / relative
        if not source.exists():
            continue
        destination = WORK / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            continue
        shell(["cp", "-r", str(source), str(destination)], step="collect",
              cwd=WORK, check=False)
    log(f"Outputs under {WORK}")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    update_status("starting", True)
    try:
        for name, function in (("checkout", checkout),
                               ("gpu_check", gpu_check),
                               ("install", install),
                               ("smoke", smoke),
                               ("splits", splits),
                               ("sample", sample),
                               ("aspects", aspects),
                               ("index", index),
                               ("eval_retrieval", eval_retrieval),
                               ("raft_build", raft_build),
                               ("raft_train", raft_train),
                               ("collect", collect_outputs)):
            run_step(name, function)
        update_status("complete", True)
        log("Job 1 finished.")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        update_status(_STATE.get("step", "unknown"), False, detail)
        log(detail)
        raise


if __name__ == "__main__":
    main()
