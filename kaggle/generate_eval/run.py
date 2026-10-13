"""
Kaggle job 2 — generation + evaluation (plan §9 notebook 2, as a script kernel).

Job 1's output is attached through `kernel_sources`, so the samples, the index
and the RAFT adapter are copied in rather than rebuilt.

  A  S1 (gemma-2-2b-it)  | S4 (RAFT + aspects)        two GPUs in parallel
  B  S2 (gemma-3-4b-it)  | S5 (RAFT, no aspects)
  B2 S6 (gemma-3, no aspects, zero-shot)              the A.5 ablation
  C  crag_tune on 150 val rows, generated with and without context
  D  S3 = the CRAG router, reusing S2 for every row above the threshold
  E  minimal_run report

Each runner checkpoints every 50 rows into its output directory, so a restarted
session resumes instead of regenerating.
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
JOB1_INPUT = Path("/kaggle/input/aspectforge-prep-train")
GENERATION_DIR = "output/generation"
SAMPLE = "data/minimal/test_500_with_aspects.csv"
VAL_SAMPLE = "data/minimal/val_500_with_aspects.csv"
ADAPTER = "output/raft/gemma3-4b-aspect-raft"
GEMMA2 = "unsloth/gemma-2-2b-it"
GEMMA3 = "unsloth/gemma-3-4b-it"
SYSTEMS = ["s1_gemma2", "s2_gemma3", "s3_gemma3_crag", "s4_raft_aspect",
           "s5_raft_noaspect", "s6_gemma3_noaspect"]

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
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step}.log"
    log(f"$ {' '.join(command)}  (-> {log_path.name})")
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n$ {' '.join(command)}\n")
        stream.flush()
        result = subprocess.run(command, cwd=str(cwd),
                                env={**os.environ, **(env or {})},
                                stdout=stream, stderr=subprocess.STDOUT,
                                text=True)
    if check and result.returncode:
        raise RuntimeError(f"{step}: command exited {result.returncode} "
                           f"(see logs/{step}.log)")
    return result.returncode


def parallel(commands: list[tuple[str, list[str], dict]]) -> None:
    """One command per GPU, both logging to their own file."""
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


def run_step(name: str, function) -> None:
    update_status(name, False)
    log(f"=== {name} ===")
    function()
    update_status(name, True)
    log(f"=== {name} done ===")


def generate(label: str, mode: str, device: str, *extra: str) -> tuple:
    command = [sys.executable, "-m", "src.generation.response_generator_v2",
               "--config", "config/generation_config.yaml",
               "--overrides", "config/kaggle.yaml",
               "--sample_csv", SAMPLE, "--mode", mode, "--label", label,
               *extra]
    return (label, command, {"CUDA_VISIBLE_DEVICES": device})


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


def install() -> None:
    shell([sys.executable, "-m", "pip", "install", "-q", "unsloth"],
          step="install")
    shell([sys.executable, "-m", "pip", "install", "-q", "-r",
           "requirements-kaggle.txt"], step="install")


def gpu_check() -> None:
    shell(["nvidia-smi"], step="gpu_check")


def copy_job1_outputs() -> None:
    """Samples, index and adapter come from job 1's kernel output."""
    if not JOB1_INPUT.exists():
        raise RuntimeError(
            f"Job 1's output is not attached at {JOB1_INPUT}; add "
            f"kernel_sources to kernel-metadata.json and push again.")
    for relative in ("data/minimal", "data/index", "output/raft",
                     "output/report"):
        source = JOB1_INPUT / relative
        if not source.exists():
            log(f"  {relative}: not in job 1's output")
            continue
        destination = CODE / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            log(f"  {relative}: already present")
            continue
        shell(["cp", "-r", str(source), str(destination)], step="copy_inputs")
    for required in (SAMPLE, "data/index/faiss_index.bin",
                     "data/index/metadata.json"):
        if not (CODE / required).exists():
            raise RuntimeError(f"Missing {required} after copying job 1's output")


def stage_a() -> None:
    parallel([
        generate("s1_gemma2", "rag_aspect", "0", "--model_id", GEMMA2),
        generate("s4_raft_aspect", "raft_aspect", "1",
                 "--model_id", GEMMA3, "--adapter_path", ADAPTER),
    ])


def stage_b() -> None:
    parallel([
        generate("s2_gemma3", "rag_aspect", "0", "--model_id", GEMMA3),
        generate("s5_raft_noaspect", "raft_rag", "1",
                 "--model_id", GEMMA3, "--adapter_path", ADAPTER),
    ])


def stage_b2() -> None:
    # S6: the same model and retrieval as S2, without aspects — this is what
    # isolates the aspect contribution without fine-tuning (plan §A.5).
    parallel([generate("s6_gemma3_noaspect", "rag_aspect", "0",
                       "--model_id", GEMMA3, "--no_aspects")])


def stage_c_crag_tune() -> None:
    """150 val rows generated twice (with and without context), then the sweep."""
    import pandas as pd

    val = pd.read_csv(CODE / VAL_SAMPLE, dtype=str).fillna("").head(150)
    tune_sample = CODE / "data/minimal/crag_tune_150.csv"
    val.to_csv(tune_sample, index=False)

    parallel([
        ("crag_rag",
         [sys.executable, "-m", "src.generation.response_generator_v2",
          "--config", "config/generation_config.yaml",
          "--overrides", "config/kaggle.yaml",
          "--sample_csv", "data/minimal/crag_tune_150.csv",
          "--mode", "rag_aspect", "--model_id", GEMMA3,
          "--label", "crag_tune_rag"],
         {"CUDA_VISIBLE_DEVICES": "0"}),
        ("crag_no_context",
         [sys.executable, "-m", "src.generation.response_generator_v2",
          "--config", "config/generation_config.yaml",
          "--overrides", "config/kaggle.yaml",
          "--sample_csv", "data/minimal/crag_tune_150.csv",
          "--mode", "rag_aspect_crag", "--model_id", GEMMA3,
          # a threshold above every possible score forces the no-context route
          "--crag_threshold", "99", "--label", "crag_tune_no_context"],
         {"CUDA_VISIBLE_DEVICES": "1"}),
    ])

    rag = pd.read_csv(CODE / GENERATION_DIR / "crag_tune_rag.csv", dtype=str)
    none = pd.read_csv(CODE / GENERATION_DIR / "crag_tune_no_context.csv",
                       dtype=str)
    paired = rag[["idx", "language", "answer", "top_ce_score"]].copy()
    paired["rag_response"] = rag["crag_tune_rag_response"]
    paired = paired.merge(
        none[["idx", "crag_tune_no_context_response"]].rename(
            columns={"crag_tune_no_context_response": "no_context_response"}),
        on="idx", how="inner")
    paired_path = CODE / "output/report/crag_pairs.csv"
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_path, index=False)

    shell([sys.executable, "-m", "src.minimal_run", "crag_tune",
           "--rows", "output/report/crag_pairs.csv",
           "--out", "output/report/crag_tuning.csv",
           "--write_config", "config/generation_config.yaml"],
          step="crag_tune")


def crag_threshold() -> float:
    import pandas as pd

    table = pd.read_csv(CODE / "output/report/crag_tuning.csv")
    best = table[table["best"].astype(str).str.lower().isin(["true", "1"])]
    return float((best if len(best) else table).iloc[0]["threshold"])


def stage_d_s3() -> None:
    """
    S3 reuses S2 for every row above the threshold (plan §7.5): identical model,
    identical prompt, so re-generating them would buy nothing. Only the rows
    below it need a no-context generation.
    """
    threshold = crag_threshold()
    log(f"CRAG threshold = {threshold}")

    import pandas as pd

    s2 = pd.read_csv(CODE / GENERATION_DIR / "s2_gemma3.csv", dtype=str).fillna("")
    scores = pd.to_numeric(s2["top_ce_score"], errors="coerce")
    below = s2[~(scores >= threshold)]
    log(f"{len(below)}/{len(s2)} rows route to no_context")

    if len(below):
        subset = pd.read_csv(CODE / SAMPLE, dtype=str).fillna("")
        subset = subset[subset["idx"].astype(str).isin(
            below["idx"].astype(str))]
        subset_path = CODE / "data/minimal/s3_below_threshold.csv"
        subset.to_csv(subset_path, index=False)
        shell([sys.executable, "-m", "src.generation.response_generator_v2",
               "--config", "config/generation_config.yaml",
               "--overrides", "config/kaggle.yaml",
               "--sample_csv", "data/minimal/s3_below_threshold.csv",
               "--mode", "rag_aspect_crag", "--model_id", GEMMA3,
               "--crag_threshold", "99", "--label", "s3_no_context"],
              step="s3_no_context", env={"CUDA_VISIBLE_DEVICES": "0"})
        no_context = f"{GENERATION_DIR}/s3_no_context.csv"
    else:
        # nothing routes to no_context: an empty file keeps crag_reuse happy
        empty = CODE / GENERATION_DIR / "s3_no_context.csv"
        empty.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["idx", "s3_no_context_prompt",
                              "s3_no_context_response"]).to_csv(empty,
                                                                index=False)
        no_context = f"{GENERATION_DIR}/s3_no_context.csv"

    shell([sys.executable, "-m", "src.minimal_run", "crag_reuse",
           "--s2", f"{GENERATION_DIR}/s2_gemma3.csv",
           "--no_context", no_context,
           "--threshold", str(threshold),
           "--label", "s3_gemma3_crag",
           "--out", f"{GENERATION_DIR}/s3_gemma3_crag.csv"],
          step="s3_reuse")


def stage_e_report() -> None:
    available = [label for label in SYSTEMS
                 if (CODE / GENERATION_DIR / f"{label}.csv").is_file()]
    log(f"Reporting on {available}")
    shell([sys.executable, "-m", "src.minimal_run", "report",
           "--systems", *available,
           "--output_dir", GENERATION_DIR,
           "--report_dir", "output/report",
           "--retrieval_config", "config/inference_retrieval_config.yaml",
           "--trainer_log", "logs/raft_train.log"], step="report")


def collect_outputs() -> None:
    for relative in (GENERATION_DIR, "output/report"):
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
                               ("copy_inputs", copy_job1_outputs),
                               ("stage_a", stage_a),
                               ("stage_b", stage_b),
                               ("stage_b2", stage_b2),
                               ("crag_tune", stage_c_crag_tune),
                               ("s3", stage_d_s3),
                               ("report", stage_e_report),
                               ("collect", collect_outputs)):
            run_step(name, function)
        update_status("complete", True)
        log("Job 2 finished.")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        update_status(_STATE.get("step", "unknown"), False, detail)
        log(detail)
        raise


if __name__ == "__main__":
    main()
