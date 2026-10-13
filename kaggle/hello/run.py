"""Minimal Kaggle GPU smoke job for the AspectForge automation scaffold."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import shutil
import time
import traceback

REPO_URL = "__ASPECTFORGE_REPO_URL__"
BRANCH = "__ASPECTFORGE_BRANCH__"
COMMIT = "__ASPECTFORGE_COMMIT__"
JOB = "__ASPECTFORGE_JOB__"
WORK = Path("/kaggle/working")
REPO_DIR = WORK / "aspectforge"
LOG_DIR = WORK / "logs"
STATUS_PATH = WORK / "STATUS.json"
LOG_PATH = LOG_DIR / f"{JOB}.log"


def update_status(step: str, ok: bool, error: str | None = None) -> None:
    payload = {
        "job": JOB,
        "step": step,
        "ok": ok,
        "error": error,
        "branch": BRANCH,
        "commit": COMMIT,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def log(message: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    line = f"[{stamp}] {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    update_status("starting", True)
    try:
        update_status("checkout", False)
        log(f"Cloning {REPO_URL} branch {BRANCH}")
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["git", "fetch", "--depth", "1", "origin", COMMIT],
            cwd=REPO_DIR, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        subprocess.run(
            ["git", "checkout", "--detach", COMMIT],
            cwd=REPO_DIR, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        actual = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        ).stdout.strip()
        if actual != COMMIT:
            raise RuntimeError(f"Requested commit {COMMIT}, checked out {actual}")
        update_status("checkout", True)

        update_status("gpu_check", False)
        log(f"Job {JOB}; repository {REPO_URL}; branch {BRANCH}; commit {COMMIT}")
        result = subprocess.run(
            ["nvidia-smi"], check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        for line in result.stdout.splitlines():
            log(line)
        update_status("gpu_check", True)
        update_status("gpu_hold", False)
        time.sleep(60)
        update_status("gpu_hold", True)
        update_status("complete", True)
        log("GPU smoke job completed successfully.")
    except Exception as exc:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR, ignore_errors=True)
        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        update_status("failed", False, detail)
        log(detail)
        raise
    finally:
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
