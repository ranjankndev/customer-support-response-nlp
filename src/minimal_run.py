"""
minimal_run.py — the fixed samples, CRAG tuning and the report (plan §6.1, §7.5, §8.2)
======================================================================================
Subcommands:

  sample          build the three fixed samples every system is scored on
  attach_aspects  merge the extractor's aspect columns back onto a sample

USAGE
    python -m src.minimal_run sample --overrides config/kaggle.yaml \
        --test_n 500 --val_n 500 --raft_n 3000 --seed 42
    python -m src.minimal_run attach_aspects \
        --sample data/minimal/test_500.csv \
        --aspects data/minimal/test_500_aspects.csv

Why fixed samples: S1-S6 have to be compared on exactly the same rows, and the
report's bootstrap is paired over row indices. A file that is silently resampled
between two runs would quietly invalidate every paired difference, so an
existing sample is always reused and its hash is printed.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

import pandas as pd

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
SAMPLE_DIR_DEFAULT = "data/minimal"
SEED_DEFAULT = 42
# columns a sample must carry into generation and scoring
CORE_COLUMNS = ["idx", "language", "subject", "body", "answer"]
ASPECT_COLUMNS = ["prob_sub", "prob_statement", "cause", "intent",
                  "ticket_type_nli", "entities", "action_taken", "urgency"]


@dataclass
class SampleSpec:
    """One sample: which split it comes from, how many rows, where it lands."""

    name     : str
    source   : str          # path to the split CSV
    n        : int
    out_path : str


@dataclass
class MinimalRunConfig:
    sample_dir   : str = SAMPLE_DIR_DEFAULT
    train_csv    : str = "data/processed/cs-support-multi-train.csv"
    validate_csv : str = "data/processed/cs-support-multi-validate.csv"
    test_csv     : str = "data/processed/cs-support-multi-test.csv"
    stratify_col : str = "language"
    seed         : int = SEED_DEFAULT
    test_n       : int = 500
    val_n        : int = 500
    raft_n       : int = 3000
    output_dir   : str = "output/report"
    systems      : List[str] = field(default_factory=list)


def load_minimal_config(path: Optional[str] = None,
                        overrides_path: Optional[str] = None
                        ) -> MinimalRunConfig:
    """
    Read the preprocessing config for the split paths, then let an overrides
    file (config/kaggle.yaml) redirect them to /kaggle/input.
    """
    config = MinimalRunConfig()
    if path and Path(path).is_file():
        from src.utils import load_yaml_with_overrides

        raw = load_yaml_with_overrides(path, overrides_path, section="minimal_run")
        base = PurePosixPath(str(raw.get("output_dir", "data/processed")))
        config.train_csv    = str(base / "cs-support-multi-train.csv")
        config.validate_csv = str(base / "cs-support-multi-validate.csv")
        config.test_csv     = str(base / "cs-support-multi-test.csv")
        split = raw.get("split", {}) or {}
        config.stratify_col = split.get("stratify_col", config.stratify_col)
        config.seed         = int(split.get("random_seed", config.seed))

    if overrides_path:
        import yaml

        data = yaml.safe_load(Path(overrides_path).read_text(encoding="utf-8")) or {}
        paths = data.get("paths") or {}
        data_dir = str(paths.get("data_dir", "") or "").strip()
        if data_dir:
            # On Kaggle the splits live in the uploaded dataset directory.
            for attribute, filename in (
                    ("train_csv", "cs-support-multi-train.csv"),
                    ("validate_csv", "cs-support-multi-validate.csv"),
                    ("test_csv", "cs-support-multi-test.csv")):
                setattr(config, attribute,
                        str(PurePosixPath(data_dir) / filename))
        work_dir = str(paths.get("work_dir", "") or "").strip()
        if work_dir:
            # POSIX paths: these land in a Kaggle job, not on this machine
            config.sample_dir = str(PurePosixPath(work_dir) / "data" / "minimal")
            config.output_dir = str(PurePosixPath(work_dir) / "output" / "report")
        section = data.get("minimal_run") or {}
        for key, value in section.items():
            if hasattr(config, key):
                setattr(config, key, value)
    return config


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING  (plan §6.1)
# ─────────────────────────────────────────────────────────────────────────────

def file_hash(path: str | Path) -> str:
    """Short sha256 of a file, printed whenever a sample is used or written."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:12]


def stratified_sample(df: pd.DataFrame, n: int, stratify_col: str = "language",
                      seed: int = SEED_DEFAULT) -> pd.DataFrame:
    """
    `n` rows split evenly across the values of `stratify_col` (250 EN + 250 DE
    for n=500), ordered by idx. A group with too few rows contributes all of
    them and the remainder is taken from the others, so the total is still `n`.
    """
    if n <= 0 or df.empty:
        return df.iloc[0:0].copy()
    if n >= len(df):
        return df.copy()
    if stratify_col not in df.columns:
        return df.sample(n=n, random_state=seed).sort_index()

    groups = {value: group for value, group in df.groupby(stratify_col)}
    order = sorted(groups)
    per_group = {value: n // len(order) for value in order}
    for i in range(n - sum(per_group.values())):      # spread the remainder
        per_group[order[i % len(order)]] += 1

    taken: Dict[str, pd.DataFrame] = {}
    short = 0
    for value in order:
        group = groups[value]
        want = per_group[value]
        if len(group) <= want:
            taken[value] = group.copy()
            short += want - len(group)
        else:
            taken[value] = group.sample(n=want, random_state=seed)
    # make up any shortfall from the groups that still have rows
    for value in order:
        if short <= 0:
            break
        group = groups[value]
        already = taken[value]
        spare = group.drop(index=already.index)
        if spare.empty:
            continue
        extra = spare.sample(n=min(short, len(spare)), random_state=seed)
        taken[value] = pd.concat([already, extra])
        short -= len(extra)

    sample = pd.concat([taken[value] for value in order])
    sort_key = "idx" if "idx" in sample.columns else None
    if sort_key:
        sample = sample.sort_values(sort_key, key=lambda s: pd.to_numeric(
            s, errors="coerce"))
    return sample.reset_index(drop=True)


def build_sample(spec: SampleSpec, stratify_col: str, seed: int,
                 force: bool = False) -> pd.DataFrame:
    """Write one sample, or reuse the existing file (never resample silently)."""
    out_path = Path(spec.out_path)
    if out_path.is_file() and not force:
        existing = pd.read_csv(out_path, dtype=str).fillna("")
        print(f"[Sample] {spec.name}: reusing {out_path} "
              f"({len(existing)} rows, sha256:{file_hash(out_path)})")
        return existing

    source = Path(spec.source)
    if not source.is_file():
        raise FileNotFoundError(
            f"{spec.name}: split not found at {source}. Build the splits first "
            f"(python -m src.preprocessing), or point --config/--overrides at "
            f"the directory that holds them."
        )
    df = pd.read_csv(source, dtype=str, low_memory=False).fillna("")
    if "idx" not in df.columns:
        df["idx"] = df.index.astype(str)
    sample = stratified_sample(df, spec.n, stratify_col, seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out_path, index=False)
    distribution = (sample[stratify_col].value_counts().to_dict()
                    if stratify_col in sample.columns else {})
    print(f"[Sample] {spec.name}: {len(sample)} rows -> {out_path} "
          f"sha256:{file_hash(out_path)} {distribution}")
    return sample


def sample_specs(config: MinimalRunConfig) -> List[SampleSpec]:
    directory = PurePosixPath(str(config.sample_dir).replace("\\", "/"))
    return [
        SampleSpec("test_500", config.test_csv, config.test_n,
                   str(directory / "test_500.csv")),
        SampleSpec("val_500", config.validate_csv, config.val_n,
                   str(directory / "val_500.csv")),
        SampleSpec(f"raft_train_{config.raft_n}", config.train_csv,
                   config.raft_n,
                   str(directory / f"raft_train_{config.raft_n}.csv")),
    ]


def cmd_sample(config: MinimalRunConfig, force: bool = False) -> Dict[str, str]:
    written = {}
    for spec in sample_specs(config):
        build_sample(spec, config.stratify_col, config.seed, force)
        written[spec.name] = spec.out_path

    # The evaluation samples must never be retrievable (plan §6.2/§7.4).
    test_ids = set(pd.read_csv(written["test_500"], dtype=str)["idx"])
    val_ids = set(pd.read_csv(written["val_500"], dtype=str)["idx"])
    overlap = test_ids & val_ids
    if overlap:
        raise AssertionError(f"test_500 and val_500 share {len(overlap)} idx "
                             f"values (e.g. {sorted(overlap)[:5]})")
    raft_name = next(name for name in written if name.startswith("raft_train"))
    raft_ids = set(pd.read_csv(written[raft_name], dtype=str)["idx"])
    leak = raft_ids & (test_ids | val_ids)
    if leak:
        raise AssertionError(f"the RAFT sample shares {len(leak)} idx values "
                             f"with the evaluation samples")
    print("[Sample] No idx overlap between test_500, val_500 and the RAFT sample")
    return written


# ─────────────────────────────────────────────────────────────────────────────
# ATTACH ASPECTS  (plan §6.1)
# ─────────────────────────────────────────────────────────────────────────────

def attach_aspects(sample_df: pd.DataFrame, aspects_df: pd.DataFrame
                   ) -> pd.DataFrame:
    """
    Merge the extractor's aspect columns onto a sample by `idx`.

    The extractor keeps idx/language/subject/body but drops `answer`, and the
    generator needs the gold answer in the same file, so the merge goes the
    other way round: the sample is the left side and only the aspect columns are
    taken from the extractor's output.
    """
    if "idx" not in sample_df.columns or "idx" not in aspects_df.columns:
        raise ValueError("both the sample and the aspects CSV need an `idx` column")

    sample = sample_df.copy()
    aspects = aspects_df.copy()
    sample["idx"] = sample["idx"].astype(str).str.strip()
    aspects["idx"] = aspects["idx"].astype(str).str.strip()

    duplicates = aspects["idx"].duplicated().sum()
    if duplicates:
        raise ValueError(f"the aspects CSV has {duplicates} duplicate idx values")

    present = [column for column in ASPECT_COLUMNS if column in aspects.columns]
    if not present:
        raise ValueError(f"the aspects CSV has none of the aspect columns "
                         f"{ASPECT_COLUMNS}; columns: {list(aspects.columns)}")

    # drop any stale aspect columns on the sample so the merge cannot suffix them
    sample = sample.drop(columns=[c for c in present if c in sample.columns])
    merged = sample.merge(aspects[["idx"] + present], on="idx", how="left")

    missing = merged[present[0]].isna().sum()
    if missing:
        print(f"[Aspects] WARNING: {missing}/{len(merged)} rows got no aspects "
              f"(idx mismatch between the sample and the extractor output)")
    merged[present] = merged[present].fillna("none")

    for column in CORE_COLUMNS:
        if column not in merged.columns:
            print(f"[Aspects] WARNING: the sample has no `{column}` column")
    return merged


def cmd_attach_aspects(sample_path: str, aspects_path: str,
                       out_path: Optional[str] = None) -> str:
    sample = pd.read_csv(sample_path, dtype=str).fillna("")
    aspects = pd.read_csv(aspects_path, dtype=str).fillna("")
    merged = attach_aspects(sample, aspects)

    destination = Path(out_path or Path(sample_path).with_name(
        Path(sample_path).stem + "_with_aspects.csv"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(destination, index=False)
    print(f"[Aspects] {len(merged)} rows -> {destination} "
          f"sha256:{file_hash(destination)}")
    print(f"[Aspects] Columns: {list(merged.columns)}")
    return str(destination)


# ─────────────────────────────────────────────────────────────────────────────
# CRAG TUNING  (plan §7.5)
# ─────────────────────────────────────────────────────────────────────────────

CRAG_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 13)]   # 0.05 .. 0.60


@dataclass
class _RougeCfg:
    rouge_types: List[str] = field(default_factory=lambda: ["rougeL"])
    rouge_unicode_tokenizer: bool = True
    rouge_use_stemmer: bool = False


def _rouge_l_rows(hypotheses: List[str], references: List[str]) -> List[float]:
    from src.utils import MetricsCalculator

    scores = MetricsCalculator(_RougeCfg()).rouge(hypotheses, references,
                                                  per_row=True)
    return [float(v) for v in scores["rougeL_rows"]]


def tune_crag_threshold(rows: pd.DataFrame,
                        thresholds: Optional[List[float]] = None
                        ) -> pd.DataFrame:
    """
    Pick the CRAG threshold without re-running generation per candidate.

    `rows` holds, per row, both outputs (`rag_response` and
    `no_context_response`), the gold `answer` and `top_ce_score`. For each τ the
    row's output is chosen as the router would choose it, and ROUGE-L is
    recomputed — so 300 generations answer twelve thresholds instead of 3600.
    """
    required = {"answer", "rag_response", "no_context_response", "top_ce_score"}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(f"crag_tune needs the columns {sorted(missing)}")

    thresholds = thresholds or CRAG_THRESHOLDS
    references = rows["answer"].fillna("").astype(str).tolist()
    rag = rows["rag_response"].fillna("").astype(str).tolist()
    no_context = rows["no_context_response"].fillna("").astype(str).tolist()
    scores = pd.to_numeric(rows["top_ce_score"], errors="coerce")

    records = []
    for threshold in thresholds:
        chosen, n_rag = [], 0
        for position, score in enumerate(scores):
            use_rag = pd.notna(score) and float(score) >= threshold
            chosen.append(rag[position] if use_rag else no_context[position])
            n_rag += int(bool(use_rag))
        per_row = _rouge_l_rows(chosen, references)
        records.append({
            "threshold": threshold,
            "rougeL": round(float(sum(per_row) / len(per_row)), 4) if per_row else 0.0,
            "rag_share": round(n_rag / len(chosen), 4) if chosen else 0.0,
            "n": len(chosen),
        })
    table = pd.DataFrame(records)
    # the smallest threshold among the best, so context is kept when it is free
    best = table[table["rougeL"] == table["rougeL"].max()]["threshold"].min()
    table["best"] = table["threshold"] == best
    return table


def cmd_crag_tune(rows_csv: str, out_path: Optional[str] = None,
                  config_path: Optional[str] = None) -> float:
    rows = pd.read_csv(rows_csv, dtype=str).fillna("")
    table = tune_crag_threshold(rows)
    print(table.to_string(index=False))

    best = float(table.loc[table["best"], "threshold"].iloc[0])
    destination = Path(out_path or "output/report/crag_tuning.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    print(f"[CRAG] Best threshold {best} -> {destination}")

    if config_path:
        write_crag_threshold(config_path, best)
    return best


def write_crag_threshold(config_path: str, threshold: float) -> None:
    """Update `crag_threshold` in generation_config.yaml in place."""
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for position, line in enumerate(lines):
        if line.split(":")[0].strip() == "crag_threshold":
            lines[position] = f"crag_threshold  : {threshold}"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[CRAG] crag_threshold = {threshold} written to {path}")
            return
    print(f"[CRAG] WARNING: no crag_threshold key in {path}; "
          f"pass --crag_threshold {threshold} instead")


def reuse_s2_outputs(s2_csv: str, no_context_csv: str, threshold: float,
                     label: str = "s3_gemma3_crag") -> pd.DataFrame:
    """
    Build S3's output file from S2's (plan §7.5): a row whose top_ce_score is at
    or above the threshold routes to "rag", and S2 already generated exactly
    that reply with exactly that prompt — only the rows below the threshold need
    the no-context generation. This halves S3's GPU cost.
    """
    s2 = pd.read_csv(s2_csv, dtype=str).fillna("")
    no_context = pd.read_csv(no_context_csv, dtype=str).fillna("")
    if "idx" not in s2.columns or "idx" not in no_context.columns:
        raise ValueError("both files need an `idx` column")

    s2_label = _response_label(s2)
    nc_label = _response_label(no_context)
    replies = dict(zip(no_context["idx"].astype(str),
                       no_context[f"{nc_label}_response"]))
    prompts = dict(zip(no_context["idx"].astype(str),
                       no_context.get(f"{nc_label}_prompt",
                                      pd.Series([""] * len(no_context)))))

    out_rows = []
    for _, row in s2.iterrows():
        idx = str(row["idx"])
        score = pd.to_numeric(pd.Series([row.get("top_ce_score", "")]),
                              errors="coerce").iloc[0]
        use_rag = pd.notna(score) and float(score) >= threshold
        if use_rag:
            reply = str(row.get(f"{s2_label}_response", "") or "")
            prompt = str(row.get(f"{s2_label}_prompt", "") or "")
            route = "rag"
        else:
            if idx not in replies:
                raise ValueError(
                    f"row {idx} routes to no_context but {no_context_csv} has "
                    f"no generation for it")
            reply = str(replies[idx] or "")
            prompt = str(prompts.get(idx, "") or "")
            route = "no_context"
        record = {column: row[column] for column in row.index
                  if not column.startswith(f"{s2_label}_")}
        record.update({f"{label}_prompt": prompt,
                       f"{label}_response": reply,
                       f"{label}_length": len(reply.split()),
                       "generated_answer": reply,
                       "route": route})
        out_rows.append(record)

    out = pd.DataFrame(out_rows)
    shares = out["route"].value_counts(normalize=True).round(4).to_dict()
    print(f"[CRAG] Reused S2 for {shares.get('rag', 0.0):.0%} of rows "
          f"(route share {shares})")
    return out


def _response_label(df: pd.DataFrame) -> str:
    """The `{label}_response` column's label."""
    for column in df.columns:
        if column.endswith("_response"):
            return column[: -len("_response")]
    raise ValueError(f"no *_response column in {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    sample_parser = commands.add_parser("sample",
                                        help="build the fixed samples")
    sample_parser.add_argument("--config", default="config/config_preprocess.yaml")
    sample_parser.add_argument("--overrides", default=None,
                               help="e.g. config/kaggle.yaml")
    sample_parser.add_argument("--sample_dir", default=None)
    sample_parser.add_argument("--test_n", type=int, default=None)
    sample_parser.add_argument("--val_n", type=int, default=None)
    sample_parser.add_argument("--raft_n", type=int, default=None)
    sample_parser.add_argument("--seed", type=int, default=None)
    sample_parser.add_argument("--force", action="store_true",
                               help="resample even if the files exist")

    attach_parser = commands.add_parser(
        "attach_aspects", help="merge aspect columns onto a sample by idx")
    attach_parser.add_argument("--sample", required=True)
    attach_parser.add_argument("--aspects", required=True)
    attach_parser.add_argument("--out", default=None)

    crag_parser = commands.add_parser(
        "crag_tune", help="pick crag_threshold from paired generations")
    crag_parser.add_argument("--rows", required=True,
                             help="CSV with answer, rag_response, "
                                  "no_context_response and top_ce_score")
    crag_parser.add_argument("--out", default="output/report/crag_tuning.csv")
    crag_parser.add_argument("--write_config", default=None,
                             help="generation config to update in place, e.g. "
                                  "config/generation_config.yaml")

    reuse_parser = commands.add_parser(
        "crag_reuse", help="build S3's output from S2's plus no-context rows")
    reuse_parser.add_argument("--s2", required=True)
    reuse_parser.add_argument("--no_context", required=True)
    reuse_parser.add_argument("--threshold", type=float, required=True)
    reuse_parser.add_argument("--label", default="s3_gemma3_crag")
    reuse_parser.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            config = load_minimal_config(args.config, args.overrides)
            for attribute in ("sample_dir", "test_n", "val_n", "raft_n", "seed"):
                value = getattr(args, attribute, None)
                if value is not None:
                    setattr(config, attribute, value)
            cmd_sample(config, force=args.force)
        elif args.command == "attach_aspects":
            cmd_attach_aspects(args.sample, args.aspects, args.out)
        elif args.command == "crag_tune":
            cmd_crag_tune(args.rows, args.out, args.write_config)
        else:
            out = reuse_s2_outputs(args.s2, args.no_context, args.threshold,
                                   args.label)
            destination = Path(args.out)
            destination.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(destination, index=False)
            print(f"[CRAG] {len(out)} rows -> {destination}")
    except (FileNotFoundError, ValueError, AssertionError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
