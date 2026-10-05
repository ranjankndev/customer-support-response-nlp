#!/usr/bin/env python3
"""Push, inspect, wait for, and download output from AspectForge Kaggle kernels."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DIR = ROOT / "kaggle"
RUNS_DIR = ROOT / "runs"
REPO_URL = "https://github.com/ranjankndev/customer-support-response-nlp.git"
SAFE_JOB = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
TERMINAL_OK = ("COMPLETE", "COMPLETED", "SUCCESS", "SUCCEEDED")
TERMINAL_BAD = ("ERROR", "FAILED", "CANCELLED", "CANCELED")


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False) -> str:
    """Run a command, forwarding output unless the caller needs to parse it."""
    try:
        result = subprocess.run(
            command, cwd=cwd, check=False, text=True, encoding="utf-8",
            errors="replace", stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Command not found: {command[0]}") from exc
    output = (result.stdout or "") if capture else ""
    if capture and output:
        print(output, end="" if output.endswith("\n") else "\n")
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}: {' '.join(command)}")
    return output


def kaggle_prefix() -> list[str]:
    """Use the Kaggle CLI installed for the active Python interpreter."""
    return [sys.executable, "-m", "kaggle"]


def read_job(job: str) -> tuple[dict, Path, Path]:
    if not SAFE_JOB.fullmatch(job):
        raise RuntimeError("Job name may contain lowercase letters, digits, '_' and '-'.")
    folder = (KAGGLE_DIR / job).resolve()
    if folder.parent != KAGGLE_DIR.resolve():
        raise RuntimeError("Job path must stay inside the repository's kaggle directory.")
    metadata_path = folder / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Missing job metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    kernel_id = metadata.get("id", "")
    if not isinstance(kernel_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+", kernel_id):
        raise RuntimeError(f"Invalid Kaggle kernel id in {metadata_path}: {kernel_id!r}")
    if metadata.get("code_file") != "run.py":
        raise RuntimeError(f"{metadata_path} must set code_file to run.py")
    return metadata, folder, metadata_path


def resolve_commit(requested: str | None) -> tuple[str, str]:
    commit = requested or run(["git", "rev-parse", "HEAD"], capture=True).strip()
    if not re.fullmatch(r"[0-9a-fA-F]{7,40}", commit):
        raise RuntimeError("--commit must be a full or abbreviated Git commit SHA.")
    full_sha = run(["git", "rev-parse", f"{commit}^{{commit}}"], capture=True).strip()
    branch = run(["git", "branch", "--show-current"], capture=True).strip()
    if not branch:
        raise RuntimeError("Cannot push a Kaggle job from a detached HEAD.")
    dirty = run(["git", "status", "--porcelain"], capture=True).strip()
    if dirty:
        raise RuntimeError(
            "Refusing to push from a dirty working tree. Commit the intended code first; "
            "Kaggle clones the GitHub branch, not local files."
        )
    return full_sha, branch


def stage_job(folder: Path, metadata_path: Path, commit: str, branch: str) -> tempfile.TemporaryDirectory:
    temp = tempfile.TemporaryDirectory(prefix="aspectforge-kaggle-")
    destination = Path(temp.name)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shutil.copy2(metadata_path, destination / metadata_path.name)
    template = folder / "run.py"
    if not template.is_file():
        temp.cleanup()
        raise RuntimeError(f"Missing job entry point: {template}")
    source = template.read_text(encoding="utf-8")
    replacements = {
        "__ASPECTFORGE_REPO_URL__": REPO_URL,
        "__ASPECTFORGE_BRANCH__": branch,
        "__ASPECTFORGE_COMMIT__": commit,
        "__ASPECTFORGE_JOB__": metadata["id"].rsplit("/", 1)[-1],
    }
    for marker, value in replacements.items():
        source = source.replace(marker, value)
    if "__ASPECTFORGE_" in source:
        temp.cleanup()
        raise RuntimeError("Unexpanded AspectForge placeholder remains in run.py")
    (destination / "run.py").write_text(source, encoding="utf-8", newline="\n")
    return temp


def push(job: str, commit_arg: str | None) -> None:
    metadata, folder, metadata_path = read_job(job)
    commit, branch = resolve_commit(commit_arg)
    print(f"Pushing {metadata['id']} from {branch}@{commit}")
    temp = stage_job(folder, metadata_path, commit, branch)
    try:
        accelerator = metadata.get("machine_shape", "NvidiaTeslaT4")
        run(kaggle_prefix() + ["kernels", "push", "--path", temp.name, "--accelerator", accelerator])
    finally:
        temp.cleanup()


def status(job: str) -> str:
    metadata, _, _ = read_job(job)
    output = run(kaggle_prefix() + ["kernels", "status", metadata["id"]], capture=True)
    return output.strip()


def wait_for_job(job: str, poll: int, timeout: int | None) -> None:
    if poll < 10:
        raise RuntimeError("--poll must be at least 10 seconds.")
    started = time.monotonic()
    previous = None
    while True:
        current = status(job)
        tokens = re.findall(r"[A-Z][A-Z_]+", current.upper())
        state = next(
            (token for token in reversed(tokens)
             if token in TERMINAL_OK + TERMINAL_BAD + ("RUNNING", "QUEUED", "PENDING", "INITIALIZED", "CANCELING")),
            current,
        )
        if state != previous:
            stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            print(f"[{stamp}] {job}: {state}")
            previous = state
        if state in TERMINAL_OK:
            return
        if state in TERMINAL_BAD:
            raise RuntimeError(f"Kaggle job {job} ended with status {state}.")
        if timeout is not None and time.monotonic() - started >= timeout:
            raise RuntimeError(f"Timed out waiting for {job} after {timeout} seconds.")
        time.sleep(poll)


def pull(job: str, out: str | None) -> None:
    metadata, _, _ = read_job(job)
    if out:
        destination = Path(out)
        if not destination.is_absolute():
            destination = ROOT / destination
    else:
        stamp = dt.datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
        destination = RUNS_DIR / job / stamp
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Downloading output from {metadata['id']} to {destination}")
    run(kaggle_prefix() + ["kernels", "output", metadata["id"], "--path", str(destination)])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("push", "status", "wait", "pull"):
        sub = commands.add_parser(name)
        sub.add_argument("--job", required=True, help="Job folder under kaggle/ (e.g. hello)")
        if name == "push":
            sub.add_argument("--commit", help="Commit SHA to bake into the Kaggle run")
        elif name == "wait":
            sub.add_argument("--poll", type=int, default=300, help="Status interval in seconds")
            sub.add_argument("--timeout", type=int, help="Optional timeout in seconds")
        elif name == "pull":
            sub.add_argument("--out", help="Output directory (default: runs/<job>/<timestamp>)")
    args = parser.parse_args()
    try:
        if args.command == "push":
            push(args.job, args.commit)
        elif args.command == "status":
            status(args.job)
        elif args.command == "wait":
            wait_for_job(args.job, args.poll, args.timeout)
        elif args.command == "pull":
            pull(args.job, args.out)
    except (RuntimeError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
