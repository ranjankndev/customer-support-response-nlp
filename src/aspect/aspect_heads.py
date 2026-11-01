"""
aspect_heads.py — Supervised aspect heads (plan A.1)
=====================================================
Replaces the SFT label model at inference. Evidence from the test sample
(docs/samples/aspects_SAMPLE.csv) is that the SFT labels collapsed: 6 of 11 rows
got `intent = general_inquiry` and `ticket_type_nli` only ever took two of four
values, while the dataset already carries much better signals as gold columns.

So the heads learn the gold columns of the **train split** instead:

    intent      <- queue       (about 10 classes)
    ticket_type <- type        (Incident / Request / Problem / Change)
    urgency     <- priority    (low / medium / high)

Features are the dense embeddings the indexer already computed for every train
ticket (`data/index/dense.npy`, row order = metadata order), so training costs
seconds on CPU and no GPU time on Kaggle. Predictions are model outputs, never
gold, so nothing leaks into evaluation: the gold columns are used only to train
and to score.

USAGE
    python -m src.aspect.aspect_heads train --config config/retrieval_config.yaml
    python -m src.aspect.aspect_heads eval  --config config/retrieval_config.yaml \
        --eval_csv output/minimal_run/val_500.csv
    python -m src.aspect.aspect_heads predict --input tickets.csv --out heads.csv
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
# head name -> gold column it is trained on (plan A.1)
HEAD_TARGETS: Dict[str, str] = {
    "intent"     : "queue",
    "ticket_type": "type",
    "urgency"    : "priority",
}
HEADS_PATH_DEFAULT = "data/index/aspect_heads.joblib"
DENSE_PATH_DEFAULT = "data/index/dense.npy"
ASPECT_TABLE_DEFAULT = "output/report/aspect_table.csv"
RANDOM_STATE = 42


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeadsConfig:
    train_csv    : str = "data/processed/cs-support-multi-train.csv"
    dense_path   : str = DENSE_PATH_DEFAULT
    metadata_path: str = "data/index/metadata.json"
    heads_path   : str = HEADS_PATH_DEFAULT
    model_name   : str = "BAAI/bge-m3"
    device       : str = "cpu"
    batch_size   : int = 32
    body_max_chars: int = 1800
    col_subject  : str = "subject"
    col_body     : str = "body"
    col_language : str = "language"
    col_ticket_id: str = "ticket_id"
    heads        : List[str] = field(default_factory=lambda: list(HEAD_TARGETS))
    max_iter     : int = 2000
    aspect_table : str = ASPECT_TABLE_DEFAULT


def load_heads_config(path: Optional[str] = None,
                      overrides_path: Optional[str] = None) -> HeadsConfig:
    """Reuse retrieval_config.yaml: the heads must embed with the same model and
    read the same dense matrix as the index."""
    if not path:
        return HeadsConfig()
    from src.utils import load_yaml_with_overrides

    cfg = load_yaml_with_overrides(path, overrides_path, section="retrieval")
    known = HeadsConfig.__dataclass_fields__
    return HeadsConfig(**{k: v for k, v in cfg.items() if k in known})


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def embedding_text(subject: str, body: str, body_max_chars: int = 0) -> str:
    """Same text the indexer embeds, so train features match query features."""
    body = str(body or "")
    snippet = body[:body_max_chars] if body_max_chars else body
    return f"{str(subject or '')}\n\n{snippet}"


def embed_texts(texts: List[str], config: HeadsConfig) -> np.ndarray:
    """Embed with the index's model and the same `passage: ` prefix."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.model_name, device=config.device)
    vecs = model.encode([f"passage: {t}" for t in texts],
                        batch_size=config.batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=len(texts) > 500)
    return np.asarray(vecs, dtype=np.float32)


def load_dense(config: HeadsConfig) -> np.ndarray:
    dense_path = Path(config.dense_path)
    if not dense_path.is_file():
        raise FileNotFoundError(
            f"Dense matrix not found: {dense_path}. Build the index first "
            f"(python -m src.retrieval.indexer --config config/retrieval_config.yaml); "
            f"it writes dense_path alongside the FAISS index."
        )
    return np.load(dense_path)


def align_targets(config: HeadsConfig, n_rows: int) -> pd.DataFrame:
    """
    Gold labels for the indexed rows, in dense.npy row order.

    metadata.json holds one entry per indexed row, so its ticket_id order defines
    the alignment; without ticket ids the train CSV is assumed to be in the same
    order as the index (which is how the indexer builds it).
    """
    train = pd.read_csv(config.train_csv, dtype=str, low_memory=False).fillna("")
    metadata_path = Path(config.metadata_path)
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        ids = [str(entry.get("ticket_id", "")) for entry in metadata]
        if (config.col_ticket_id in train.columns
                and set(ids) <= set(train[config.col_ticket_id].astype(str))):
            indexed = train.set_index(train[config.col_ticket_id].astype(str))
            return indexed.loc[ids].reset_index(drop=True)
        if len(metadata) != n_rows:
            raise ValueError(f"metadata.json has {len(metadata)} rows but "
                             f"dense matrix has {n_rows}")
    if len(train) < n_rows:
        raise ValueError(f"{config.train_csv} has {len(train)} rows, "
                         f"dense matrix has {n_rows}")
    return train.iloc[:n_rows].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def train_heads(features: np.ndarray, targets: pd.DataFrame,
                heads: Optional[List[str]] = None,
                max_iter: int = 2000) -> Dict[str, object]:
    """
    One balanced logistic regression per head. Returns a bundle that also records
    the majority class per head, so `eval` can always report the baseline the
    heads have to beat.
    """
    from sklearn.linear_model import LogisticRegression

    heads = heads or list(HEAD_TARGETS)
    models: Dict[str, object] = {}
    majority: Dict[str, str] = {}
    for head in heads:
        column = HEAD_TARGETS[head]
        if column not in targets.columns:
            raise ValueError(f"Gold column '{column}' for head '{head}' is missing "
                             f"from the train CSV; available: {list(targets.columns)}")
        labels = targets[column].astype(str).str.strip()
        keep = labels.ne("") & labels.str.lower().ne("nan")
        if keep.sum() < 2 or labels[keep].nunique() < 2:
            raise ValueError(f"Head '{head}' needs at least two classes, "
                             f"got {labels[keep].nunique()}")
        model = LogisticRegression(max_iter=max_iter, class_weight="balanced")
        model.fit(features[keep.values], labels[keep].values)
        models[head] = model
        majority[head] = labels[keep].value_counts().idxmax()

    return {"heads": models, "majority": majority,
            "targets": {h: HEAD_TARGETS[h] for h in heads},
            "dim": int(features.shape[1])}


def save_bundle(bundle: Dict[str, object], path: str) -> None:
    import joblib

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    print(f"[Heads] Saved -> {out}")


def load_bundle(path: str) -> Dict[str, object]:
    import joblib

    bundle_path = Path(path)
    if not bundle_path.is_file():
        raise FileNotFoundError(
            f"Aspect heads not found: {bundle_path}. Train them first: "
            f"python -m src.aspect.aspect_heads train --config config/retrieval_config.yaml"
        )
    return joblib.load(bundle_path)


# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────

def predict_from_features(bundle: Dict[str, object],
                          features: np.ndarray) -> Dict[str, List[str]]:
    dim = int(bundle.get("dim", features.shape[1]))
    if features.shape[1] != dim:
        raise ValueError(f"Heads were trained on {dim}-dimensional embeddings, "
                         f"got {features.shape[1]}: embed with the same model.")
    return {head: [str(v) for v in model.predict(features)]
            for head, model in bundle["heads"].items()}


def predict_dataframe(df: pd.DataFrame, config: HeadsConfig,
                      bundle: Optional[Dict[str, object]] = None) -> pd.DataFrame:
    """Add one column per head to `df` (CPU: no SFT model is loaded)."""
    bundle = bundle or load_bundle(config.heads_path)
    texts = [embedding_text(row.get(config.col_subject, ""),
                            row.get(config.col_body, ""),
                            config.body_max_chars)
             for _, row in df.iterrows()]
    predictions = predict_from_features(bundle, embed_texts(texts, config))
    out = df.copy()
    for head, values in predictions.items():
        out[head] = values
    # the pipeline's legacy column name for the ticket type
    if "ticket_type" in out.columns:
        out["ticket_type_nli"] = out["ticket_type"]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# EVAL
# ─────────────────────────────────────────────────────────────────────────────

def score_head(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro",
                                             zero_division=0)), 4),
            "n": len(y_true)}


def evaluate_heads(bundle: Dict[str, object], features: np.ndarray,
                   targets: pd.DataFrame,
                   languages: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Accuracy and macro-F1 per head, overall and per language, each next to the
    majority-class baseline the head has to beat.
    """
    predictions = predict_from_features(bundle, features)
    majority = bundle.get("majority", {})
    rows = []
    for head, y_pred in predictions.items():
        column = bundle["targets"][head]
        y_true = targets[column].astype(str).str.strip().tolist()
        groups = [("all", np.ones(len(y_true), dtype=bool))]
        if languages is not None:
            codes = languages.astype(str).str.lower().str[:2]
            groups += [(lang, (codes == lang).values)
                       for lang in sorted(codes.unique()) if lang]
        for name, mask in groups:
            true_subset = [t for t, keep in zip(y_true, mask) if keep]
            pred_subset = [p for p, keep in zip(y_pred, mask) if keep]
            if len(true_subset) < 2:
                continue
            head_scores = score_head(true_subset, pred_subset)
            base_label = majority.get(head, "")
            base_scores = score_head(true_subset, [base_label] * len(true_subset))
            rows.append({
                "head": head, "gold_column": column, "split": name,
                "n": head_scores["n"],
                "accuracy": head_scores["accuracy"],
                "f1_macro": head_scores["f1_macro"],
                "majority_label": base_label,
                "majority_accuracy": base_scores["accuracy"],
                "majority_f1_macro": base_scores["f1_macro"],
                "beats_majority": bool(head_scores["f1_macro"]
                                       > base_scores["f1_macro"]),
            })
    return pd.DataFrame(rows)


def sft_label_collapse(aspects_csv: str) -> pd.DataFrame:
    """
    Document the collapse the heads replace (plan A.1): share of
    `general_inquiry` and the number of distinct values the SFT labels took.
    """
    df = pd.read_csv(aspects_csv, dtype=str).fillna("")
    rows = []
    for column, collapsed in (("intent", "general_inquiry"),
                              ("ticket_type_nli", "")):
        if column not in df.columns:
            continue
        values = df[column].astype(str).str.strip().str.lower()
        rows.append({
            "label": column,
            "n": len(values),
            "distinct_values": int(values.nunique()),
            "top_value": values.value_counts().idxmax() if len(values) else "",
            "top_share": round(float(values.value_counts(normalize=True).iloc[0]), 4)
                         if len(values) else 0.0,
            "collapsed_share": (round(float(values.eq(collapsed).mean()), 4)
                                if collapsed else None),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_train(args) -> None:
    config = load_heads_config(args.config, args.overrides)
    if args.heads_path:
        config.heads_path = args.heads_path
    features = load_dense(config)
    targets = align_targets(config, features.shape[0])
    print(f"[Heads] Training on {features.shape[0]} rows x {features.shape[1]} dims")
    bundle = train_heads(features, targets, config.heads, config.max_iter)
    save_bundle(bundle, config.heads_path)
    for head, label in bundle["majority"].items():
        print(f"[Heads] {head:<12} <- {bundle['targets'][head]:<10} "
              f"majority={label}")


def _cmd_eval(args) -> None:
    config = load_heads_config(args.config, args.overrides)
    if args.heads_path:
        config.heads_path = args.heads_path
    bundle = load_bundle(config.heads_path)
    df = pd.read_csv(args.eval_csv, dtype=str, low_memory=False).fillna("")
    texts = [embedding_text(row.get(config.col_subject, ""),
                            row.get(config.col_body, ""),
                            config.body_max_chars)
             for _, row in df.iterrows()]
    features = embed_texts(texts, config)
    languages = df[config.col_language] if config.col_language in df.columns else None
    table = evaluate_heads(bundle, features, df, languages)
    print(table.to_string(index=False))

    out_path = Path(args.out or config.aspect_table)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [table.assign(source="heads")]
    if args.sft_aspects_csv:
        collapse = sft_label_collapse(args.sft_aspects_csv)
        print("\n[Heads] SFT label distribution (what the heads replace):")
        print(collapse.to_string(index=False))
        frames.append(collapse.assign(source="sft_labels"))
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"\n[Heads] Aspect table -> {out_path}")


def _cmd_predict(args) -> None:
    config = load_heads_config(args.config, args.overrides)
    if args.heads_path:
        config.heads_path = args.heads_path
    df = pd.read_csv(args.input, dtype=str, low_memory=False).fillna("")
    if args.n:
        df = df.head(args.n)
    out = predict_dataframe(df, config)
    out_path = Path(args.out or "aspect_heads_predictions.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[Heads] Predicted {len(out)} rows -> {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval", "predict"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="config/retrieval_config.yaml",
                         help="retrieval config (embedder + dense_path)")
        sub.add_argument("--overrides", default=None,
                         help="YAML whose `retrieval` section is merged on top")
        sub.add_argument("--heads_path", default=None,
                         help=f"Head bundle (default: {HEADS_PATH_DEFAULT})")
        if name == "eval":
            sub.add_argument("--eval_csv", required=True,
                             help="CSV with subject, body and the gold columns")
            sub.add_argument("--sft_aspects_csv", default=None,
                             help="Old SFT aspects CSV, to document the collapse")
            sub.add_argument("--out", default=None,
                             help=f"Aspect table (default: {ASPECT_TABLE_DEFAULT})")
        elif name == "predict":
            sub.add_argument("--input", required=True)
            sub.add_argument("--out", default=None)
            sub.add_argument("--n", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        if args.command == "train":
            _cmd_train(args)
        elif args.command == "eval":
            _cmd_eval(args)
        else:
            _cmd_predict(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
