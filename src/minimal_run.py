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
# REPORT  (plan §8.2 and §A.4)
# ─────────────────────────────────────────────────────────────────────────────

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
# metrics shown in the table, in order; ACS/AAS come from §A.4
REPORT_METRICS = ["rouge1", "rouge2", "rougeL", "bertscore_f1", "lang_match",
                  "acs", "aas"]
BASELINE_SYSTEM = "s2_gemma3"


def load_systems(output_dir: str, systems: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Load {output_dir}/{label}.csv per system and align them on a common idx.

    Every system must cover the same rows: the bootstrap below is paired, so a
    system missing rows would be compared against a different sample.
    """
    frames = {}
    for label in systems:
        path = Path(output_dir) / f"{label}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"System output not found: {path}")
        df = pd.read_csv(path, dtype=str).fillna("")
        if "idx" not in df.columns:
            raise ValueError(f"{path} has no idx column")
        df["idx"] = df["idx"].astype(str).str.strip()
        duplicates = df["idx"].duplicated().sum()
        if duplicates:
            raise ValueError(f"{path} has {duplicates} duplicate idx values")
        frames[label] = df.set_index("idx")

    index_sets = {label: set(df.index) for label, df in frames.items()}
    shared = set.intersection(*index_sets.values())
    for label, ids in index_sets.items():
        extra = ids - shared
        if extra:
            raise AssertionError(
                f"{label} has {len(extra)} rows the other systems do not "
                f"(e.g. {sorted(extra)[:5]}): every system must be scored on "
                f"the same sample."
            )
    order = sorted(shared, key=lambda value: (len(value), value))
    print(f"[Report] {len(frames)} systems aligned on {len(order)} rows")
    return {label: df.loc[order].reset_index() for label, df in frames.items()}


def system_reply(df: pd.DataFrame, label: str) -> List[str]:
    """The system's reply column, by label, with the old names as fallbacks."""
    for column in (f"{label}_response", "generated_answer"):
        if column in df.columns:
            return df[column].fillna("").astype(str).tolist()
    candidates = [c for c in df.columns if c.endswith("_response")]
    if candidates:
        return df[candidates[0]].fillna("").astype(str).tolist()
    raise ValueError(f"{label}: no reply column in {list(df.columns)}")


def split_sentences_for_acs(text: str) -> List[str]:
    from src.aspect.aspect_extractor import split_sentences

    sentences = split_sentences(text or "")
    return sentences or ([text.strip()] if (text or "").strip() else [])


def aspect_coverage(embedder, replies: List[str], aspect_rows: List[Dict],
                    fields: List[str]) -> List[Optional[float]]:
    """
    ACS/AAS (plan §A.4): per row, the mean over its non-'none' aspects of the
    best cosine between the aspect text and any sentence of the reply.

    This is the project's own claim made measurable — ROUGE cannot tell whether
    a fluent reply actually addresses the problem it was given. A row with no
    usable aspect yields None and is left out of the mean.
    """
    scores: List[Optional[float]] = []
    for reply, aspects in zip(replies, aspect_rows):
        wanted = [str(aspects.get(field, "") or "").strip()
                  for field in fields]
        wanted = [text for text in wanted
                  if text and text.lower() not in ("none", "nan")]
        sentences = split_sentences_for_acs(reply)
        if not wanted or not sentences:
            scores.append(None)
            continue
        vectors = embedder.encode(wanted + sentences, kind="passage")
        import numpy as np

        matrix = np.asarray(vectors, dtype=np.float32)
        aspect_vectors = matrix[:len(wanted)]
        sentence_vectors = matrix[len(wanted):]
        norms = (np.linalg.norm(aspect_vectors, axis=1, keepdims=True)
                 * np.linalg.norm(sentence_vectors, axis=1, keepdims=True).T)
        similarity = np.divide(aspect_vectors @ sentence_vectors.T, norms,
                               out=np.zeros_like(norms), where=norms > 0)
        scores.append(float(similarity.max(axis=1).mean()))
    return scores


def per_row_metrics(frames: Dict[str, pd.DataFrame], calc, embedder=None,
                    with_bertscore: bool = True) -> pd.DataFrame:
    """One row per (system, idx) with every per-row metric (plan §8.2 step 2)."""
    from src.minimal_run import ASPECT_COLUMNS

    records = []
    for label, df in frames.items():
        replies = system_reply(df, label)
        references = df["answer"].fillna("").astype(str).tolist()
        languages = df["language"].fillna("").astype(str).str.lower().str[:2] \
            if "language" in df.columns else pd.Series([""] * len(df))

        rouge = calc.rouge(replies, references, per_row=True)
        lang = calc.lang_match(replies, languages.tolist(), per_row=True)
        bertscore = ({} if not with_bertscore
                     else calc.bertscore_true(replies, references, per_row=True))

        aspect_rows = [{field: row.get(field, "") for field in ASPECT_COLUMNS}
                       for _, row in df.iterrows()]
        acs = aas = [None] * len(df)
        if embedder is not None:
            acs = aspect_coverage(embedder, replies, aspect_rows,
                                  ["prob_sub", "cause"])
            aas = aspect_coverage(embedder, replies, aspect_rows,
                                  ["action_taken"])

        for position in range(len(df)):
            record = {
                "system": label,
                "idx": df["idx"].iloc[position] if "idx" in df.columns else position,
                "language": languages.iloc[position] if len(languages) else "",
                "rouge1": rouge["rouge1_rows"][position],
                "rouge2": rouge["rouge2_rows"][position],
                "rougeL": rouge["rougeL_rows"][position],
                "lang_match": lang["lang_match_rows"][position],
                "acs": acs[position],
                "aas": aas[position],
                "route": df["route"].iloc[position] if "route" in df.columns else "",
                "top_ce_score": (df["top_ce_score"].iloc[position]
                                 if "top_ce_score" in df.columns else ""),
            }
            if bertscore:
                record["bertscore_f1"] = bertscore["bertscore_f1_rows"][position]
            records.append(record)
    return pd.DataFrame(records)


def _resample_indices(n: int, resamples: int, seed: int):
    import numpy as np

    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def bootstrap_table(per_row: pd.DataFrame,
                    metrics: Optional[List[str]] = None,
                    resamples: int = BOOTSTRAP_RESAMPLES,
                    seed: int = BOOTSTRAP_SEED) -> pd.DataFrame:
    """
    Mean and 95% CI per system / split / metric, and the paired difference
    against the baseline system.

    The same resample indices are used for every system, which is what makes a
    difference paired: both systems are re-scored on the same rows, so the CI
    reflects the difference and not the sample.
    """
    import numpy as np

    metrics = metrics or [m for m in REPORT_METRICS if m in per_row.columns]
    systems = list(dict.fromkeys(per_row["system"]))
    splits = ["all"] + sorted({lang for lang in per_row["language"] if lang})
    baseline = BASELINE_SYSTEM if BASELINE_SYSTEM in systems else systems[0]

    # idx order shared by all systems, so row i means the same ticket everywhere
    order = list(dict.fromkeys(per_row[per_row["system"] == baseline]["idx"]))
    draws_cache: Dict[int, object] = {}

    def values(system: str, split: str, metric: str):
        frame = per_row[per_row["system"] == system].set_index("idx")
        frame = frame.loc[[i for i in order if i in frame.index]]
        if split != "all":
            frame = frame[frame["language"] == split]
        series = pd.to_numeric(frame[metric], errors="coerce")
        return series.to_numpy(dtype=float), list(frame.index)

    rows = []
    for split in splits:
        for metric in metrics:
            base_values, base_index = values(baseline, split, metric)
            mask = ~np.isnan(base_values)
            n = int(mask.sum())
            if n < 2:
                continue
            if n not in draws_cache:
                draws_cache[n] = _resample_indices(n, resamples, seed)
            draws = draws_cache[n]

            for system in systems:
                system_values, _ = values(system, split, metric)
                paired = system_values[mask]
                if np.isnan(paired).all():
                    continue
                clean = np.nan_to_num(paired, nan=float(np.nanmean(paired)))
                means = clean[draws].mean(axis=1)
                record = {
                    "system": system, "split": split, "metric": metric,
                    "n": n,
                    "mean": round(float(clean.mean()), 4),
                    "ci_low": round(float(np.percentile(means, 2.5)), 4),
                    "ci_high": round(float(np.percentile(means, 97.5)), 4),
                }
                if system != baseline:
                    base_clean = np.nan_to_num(
                        base_values[mask], nan=float(np.nanmean(base_values[mask])))
                    difference = clean - base_clean
                    diff_means = difference[draws].mean(axis=1)
                    low = float(np.percentile(diff_means, 2.5))
                    high = float(np.percentile(diff_means, 97.5))
                    record.update({
                        "vs_baseline": baseline,
                        "diff": round(float(difference.mean()), 4),
                        "diff_ci_low": round(low, 4),
                        "diff_ci_high": round(high, 4),
                        # significant = the CI of the paired difference misses 0
                        "significant": bool(low > 0 or high < 0),
                    })
                rows.append(record)
    return pd.DataFrame(rows)


def format_interval(row) -> str:
    return f"{row['mean']:.4f} [{row['ci_low']:.4f}, {row['ci_high']:.4f}]"


def markdown_table(table: pd.DataFrame, split: str = "all") -> str:
    """`mean [low, high]` per system and metric, for results.md."""
    subset = table[table["split"] == split]
    if subset.empty:
        return "_(no rows)_"
    metrics = [m for m in REPORT_METRICS if m in set(subset["metric"])]
    systems = list(dict.fromkeys(subset["system"]))

    lines = ["| System | " + " | ".join(metrics) + " |",
             "|---" * (len(metrics) + 1) + "|"]
    for system in systems:
        cells = []
        for metric in metrics:
            match = subset[(subset["system"] == system)
                           & (subset["metric"] == metric)]
            cells.append(format_interval(match.iloc[0]) if len(match) else "—")
        lines.append(f"| {system} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def takeaway_lines(table: pd.DataFrame) -> List[str]:
    """One sentence per significant paired difference, generated from the CIs."""
    lines = []
    if "significant" not in table.columns:
        # only one system: there is nothing to compare it against
        return ["- Only one system in this report, so there is no paired "
                "difference to test."]
    flags = table["significant"].astype("object").fillna(False).astype(bool)
    differences = table[flags]
    for _, row in differences[differences["split"] == "all"].iterrows():
        direction = "above" if row["diff"] > 0 else "below"
        lines.append(
            f"- **{row['system']}** is {abs(row['diff']):.4f} {direction} "
            f"{row['vs_baseline']} on {row['metric']} "
            f"(95% CI [{row['diff_ci_low']:.4f}, {row['diff_ci_high']:.4f}], "
            f"n={int(row['n'])}) — significant.")
    if not lines:
        lines.append("- No paired difference is significant at this sample "
                     "size: every CI includes 0.")
    return lines


def aspect_contribution(table: pd.DataFrame,
                        pairs=(("s2_gemma3", "s6_gemma3_noaspect"),
                               ("s4_raft_aspect", "s5_raft_noaspect"))
                        ) -> List[str]:
    """
    The aspect claim, stated from the numbers (plan §A.5): S2-S6 is the effect
    without fine-tuning, S4-S5 the effect with it.
    """
    lines = []
    for with_aspects, without in pairs:
        rows = table[(table["system"] == with_aspects)
                     & (table.get("vs_baseline", "") == without)
                     & (table["split"] == "all")]
        if rows.empty:
            # the table is anchored on the baseline, so derive the difference
            rows = table[(table["system"].isin([with_aspects, without]))
                         & (table["split"] == "all")]
            if set(rows["system"]) != {with_aspects, without}:
                continue
            for metric in ("rougeL", "bertscore_f1", "acs"):
                subset = rows[rows["metric"] == metric]
                if len(subset) != 2:
                    continue
                a = subset[subset["system"] == with_aspects]["mean"].iloc[0]
                b = subset[subset["system"] == without]["mean"].iloc[0]
                lines.append(f"- {with_aspects} − {without} on {metric}: "
                             f"{a - b:+.4f} ({a:.4f} vs {b:.4f}).")
            continue
        for _, row in rows.iterrows():
            if row["metric"] not in ("rougeL", "bertscore_f1", "acs"):
                continue
            verdict = "significant" if row.get("significant") else "not significant"
            lines.append(
                f"- {with_aspects} − {without} on {row['metric']}: "
                f"{row['diff']:+.4f} (95% CI [{row['diff_ci_low']:.4f}, "
                f"{row['diff_ci_high']:.4f}], {verdict}).")
    return lines


def aspect_table(table: pd.DataFrame) -> str:
    """ACS/AAS per system and language (plan §A.4)."""
    subset = table[table["metric"].isin(["acs", "aas"])]
    if subset.empty:
        return "_(no aspect scores: the report ran without an embedder)_"
    splits = ["all"] + sorted({s for s in subset["split"] if s != "all"})
    systems = list(dict.fromkeys(subset["system"]))

    header = ["| System |"]
    for metric in ("acs", "aas"):
        for split in splits:
            header.append(f" {metric.upper()} ({split}) |")
    lines = ["".join(header), "|---" * (1 + 2 * len(splits)) + "|"]
    for system in systems:
        cells = [f"| {system} |"]
        for metric in ("acs", "aas"):
            for split in splits:
                match = subset[(subset["system"] == system)
                               & (subset["metric"] == metric)
                               & (subset["split"] == split)]
                cells.append(f" {format_interval(match.iloc[0])} |"
                             if len(match) else " — |")
        lines.append("".join(cells))
    return "\n".join(lines)


def route_shares(per_row: pd.DataFrame) -> str:
    """CRAG route share per system, for results.md."""
    if "route" not in per_row.columns:
        return ""
    routed = per_row[per_row["route"].astype(str).str.strip() != ""]
    if routed.empty:
        return ""
    lines = []
    for system, frame in routed.groupby("system"):
        shares = frame["route"].value_counts(normalize=True).round(4).to_dict()
        lines.append(f"- `{system}`: " + ", ".join(
            f"{route} {share:.0%}" for route, share in sorted(shares.items())))
    return "\n".join(lines)


def qualitative_examples(frames: Dict[str, pd.DataFrame], n_en: int = 1,
                         n_de: int = 2,
                         systems: Optional[List[str]] = None) -> str:
    """Ticket, gold, and each system's reply for a few rows (plan §8.2 step 7)."""
    labels = systems or list(frames)
    first = frames[labels[0]]
    chosen = []
    for language, count in (("en", n_en), ("de", n_de)):
        matching = first[first.get("language", "").astype(str).str.lower()
                         .str[:2] == language]
        chosen.extend(matching.head(count)["idx"].tolist())

    blocks = []
    for idx in chosen:
        row = first[first["idx"] == idx].iloc[0]
        block = [f"#### Ticket `{idx}` ({row.get('language', '')})",
                 f"**Subject:** {row.get('subject', '')}",
                 "", f"**Gold answer:** {row.get('answer', '')}", ""]
        for label in labels:
            frame = frames[label]
            match = frame[frame["idx"] == idx]
            if match.empty:
                continue
            reply = system_reply(match, label)[0]
            block.append(f"**{label}:** {reply}")
            block.append("")
        blocks.append("\n".join(block))
    return "\n".join(blocks)


def build_results_markdown(table: pd.DataFrame, per_row: pd.DataFrame,
                           frames: Dict[str, pd.DataFrame],
                           retrieval_table: Optional[pd.DataFrame] = None,
                           aspect_heads_table: Optional[pd.DataFrame] = None,
                           trainer_log: str = "") -> str:
    sections = ["# AspectForge v3 — minimal run results", "",
                f"Systems: {', '.join(dict.fromkeys(table['system']))}  ",
                f"Rows per system: {int(table['n'].max())}  ",
                f"Bootstrap: {BOOTSTRAP_RESAMPLES} paired resamples, "
                f"seed {BOOTSTRAP_SEED}; a difference is called significant "
                f"when the CI of the paired difference excludes 0.", "",
                "## Overall", "", markdown_table(table, "all"), ""]

    for split in sorted({s for s in table["split"] if s != "all"}):
        sections += [f"## {split.upper()}", "", markdown_table(table, split), ""]

    sections += ["## Takeaways", ""] + takeaway_lines(table) + [""]

    contribution = aspect_contribution(table)
    if contribution:
        sections += ["## Aspect contribution", "",
                     "S2 − S6 isolates the aspects without fine-tuning, "
                     "S4 − S5 with it.", ""] + contribution + [""]

    sections += ["## Aspect coverage (ACS / AAS)", "", aspect_table(table), ""]

    routes = route_shares(per_row)
    if routes:
        sections += ["## CRAG routes", "", routes, ""]

    if retrieval_table is not None and not retrieval_table.empty:
        sections += ["## Retrieval", "",
                     retrieval_table.to_markdown(index=False), ""]
    if aspect_heads_table is not None and not aspect_heads_table.empty:
        sections += ["## Aspect heads", "",
                     aspect_heads_table.to_markdown(index=False), ""]
    if trainer_log:
        sections += ["## RAFT training", "", "```", trainer_log.strip(), "```", ""]

    sections += ["## Examples", "", qualitative_examples(frames), ""]
    return "\n".join(sections)


def results_chart(table: pd.DataFrame, path: str) -> Optional[str]:
    """Grouped bars with CI error bars per system, split by language (§8.2.6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [m for m in ("rougeL", "bertscore_f1", "lang_match")
               if m in set(table["metric"])]
    splits = [s for s in ("en", "de") if s in set(table["split"])] or ["all"]
    systems = list(dict.fromkeys(table["system"]))
    if not metrics or not systems:
        return None

    figure, axes = plt.subplots(1, len(metrics),
                                figsize=(5 * len(metrics), 4), squeeze=False)
    width = 0.8 / max(len(systems), 1)
    for column, metric in enumerate(metrics):
        axis = axes[0][column]
        positions = np.arange(len(splits))
        for offset, system in enumerate(systems):
            means, errors = [], []
            for split in splits:
                match = table[(table["system"] == system)
                              & (table["split"] == split)
                              & (table["metric"] == metric)]
                if match.empty:
                    means.append(0.0)
                    errors.append(0.0)
                    continue
                row = match.iloc[0]
                means.append(row["mean"])
                errors.append(max(row["ci_high"] - row["mean"],
                                  row["mean"] - row["ci_low"]))
            axis.bar(positions + offset * width, means, width, yerr=errors,
                     capsize=3, label=system)
        axis.set_xticks(positions + width * (len(systems) - 1) / 2)
        axis.set_xticklabels([s.upper() for s in splits])
        axis.set_title(metric)
        axis.set_ylim(bottom=0)
    axes[0][-1].legend(fontsize="small")
    figure.tight_layout()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)
    print(f"[Report] Chart -> {out}")
    return str(out)


def _metrics_calculator(bertscore_model: str = "xlm-roberta-large",
                        bertscore_batch: int = 16):
    from src.utils import MetricsCalculator

    @dataclass
    class _Cfg:
        rouge_types: List[str] = field(
            default_factory=lambda: ["rouge1", "rouge2", "rougeL"])
        rouge_unicode_tokenizer: bool = True
        rouge_use_stemmer: bool = False
        bleu_tokenizer: str = "13a"
        bertscore_true_model: str = bertscore_model
        bertscore_true_batch: int = bertscore_batch

    return MetricsCalculator(_Cfg())


def cmd_report(output_dir: str, systems: List[str], report_dir: str,
               retrieval_config: Optional[str] = None,
               with_bertscore: bool = True, with_aspects: bool = True,
               retrieval_table_path: Optional[str] = None,
               aspect_table_path: Optional[str] = None,
               trainer_log_path: Optional[str] = None) -> Dict[str, str]:
    frames = load_systems(output_dir, systems)
    calc = _metrics_calculator()

    embedder = None
    if with_aspects and retrieval_config:
        from src.retrieval.inference_retriever import load_retriever_config
        from src.retrieval.retriever import STEmbedder

        embedder = STEmbedder(load_retriever_config(retrieval_config))

    per_row = per_row_metrics(frames, calc, embedder, with_bertscore)
    table = bootstrap_table(per_row)

    destination = Path(report_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = {}

    per_row_path = destination / "per_row_metrics.csv"
    per_row.to_csv(per_row_path, index=False)
    written["per_row_metrics"] = str(per_row_path)

    table_path = destination / "results_table.csv"
    table.to_csv(table_path, index=False)
    written["results_table"] = str(table_path)

    retrieval = (pd.read_csv(retrieval_table_path)
                 if retrieval_table_path and Path(retrieval_table_path).is_file()
                 else None)
    heads = (pd.read_csv(aspect_table_path)
             if aspect_table_path and Path(aspect_table_path).is_file()
             else None)
    trainer_log = (Path(trainer_log_path).read_text(encoding="utf-8")[-4000:]
                   if trainer_log_path and Path(trainer_log_path).is_file()
                   else "")

    markdown = build_results_markdown(table, per_row, frames, retrieval, heads,
                                      trainer_log)
    markdown_path = destination / "results.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    written["results_md"] = str(markdown_path)

    chart = results_chart(table, str(destination / "results_chart.png"))
    if chart:
        written["results_chart"] = chart

    print(f"[Report] Written: {written}")
    return written


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

    report_parser = commands.add_parser(
        "report", help="metrics, bootstrap CIs, results.md and the chart")
    report_parser.add_argument("--systems", nargs="+", required=True,
                               help="labels, e.g. s1_gemma2 s2_gemma3 "
                                    "s3_gemma3_crag s4_raft_aspect "
                                    "s5_raft_noaspect")
    report_parser.add_argument("--output_dir", default="output/generation",
                               help="where the {label}.csv files are")
    report_parser.add_argument("--report_dir", default="output/report")
    report_parser.add_argument("--retrieval_config",
                               default="config/inference_retrieval_config.yaml",
                               help="embedder used for ACS/AAS")
    report_parser.add_argument("--no_bertscore", action="store_true",
                               help="skip true BERTScore (2 GB download)")
    report_parser.add_argument("--no_aspect_scores", action="store_true",
                               help="skip ACS/AAS (no embedder needed)")
    report_parser.add_argument("--retrieval_table",
                               default="output/report/retrieval_table.csv")
    report_parser.add_argument("--aspect_table",
                               default="output/report/aspect_table.csv")
    report_parser.add_argument("--trainer_log", default=None)

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
        elif args.command == "report":
            cmd_report(args.output_dir, args.systems, args.report_dir,
                       args.retrieval_config,
                       with_bertscore=not args.no_bertscore,
                       with_aspects=not args.no_aspect_scores,
                       retrieval_table_path=args.retrieval_table,
                       aspect_table_path=args.aspect_table,
                       trainer_log_path=args.trainer_log)
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
