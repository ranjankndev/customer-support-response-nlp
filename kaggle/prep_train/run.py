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
RAW_DIR = Path("/kaggle/input/multilingual-customer-support-tickets")
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


def find_raw_csv() -> Path:
    """
    The raw dataset ships several CSVs; the one that reproduces the report's
    28,237 EN+DE rows is the widest multilingual file. Pick the candidate with
    the most EN+DE rows and record the choice in the log.
    """
    import pandas as pd

    candidates = sorted(RAW_DIR.rglob("*.csv"))
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
                 "--adapter", "ranjan56cse/cs-support-labels",
                 "--adapter_subfolder", "sft_checkpoint"],
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
           "--label", "m3_hybrid_rerank"], step="eval_retrieval")


def raft_build() -> None:
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
