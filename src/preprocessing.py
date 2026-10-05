"""
preprocessing.py — Stratified Split
=====================================
Steps:
  1. Load raw CSV
  2. Add `idx` column (integer, starts from 0) as first column
  3. Bring `language` as second column
  4. Stratified split by language → train / validate / test
  5. Save three CSV files
"""

import yaml
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config_preprocess.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN REORDER
# ─────────────────────────────────────────────────────────────────────────────

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `idx` as first column (0-based integer).
    Move `language` to second column.
    All other columns follow in their original order.
    """
    df = df.copy()
    df.insert(0, "idx", range(len(df)))

    # move language to position 1 (after idx)
    cols = list(df.columns)
    if "language" in cols:
        cols.remove("language")
        cols.insert(1, "language")
    df = df[cols]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STRATIFIED SPLITTER
# ─────────────────────────────────────────────────────────────────────────────

class StratifiedSplitter:
    """
    Splits dataframe into train / validate / test
    with stratification on the language column.
    """

    def __init__(self, config: dict):
        split_cfg        = config.get("split", {})
        self.test_size   : int = split_cfg.get("test_size",     5420)
        self.val_size    : int = split_cfg.get("validate_size", 5420)
        self.strat_col   : str = split_cfg.get("stratify_col",  "language")
        self.random_seed : int = split_cfg.get("random_seed",   42)

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Returns (train_df, validate_df, test_df).
        """
        total = len(df)
        strat = df[self.strat_col] if self.strat_col in df.columns else None

        # Step 1 — carve out test set
        remaining, test_df = train_test_split(
            df,
            test_size    = self.test_size / total,
            stratify     = strat,
            random_state = self.random_seed,
        )

        # Step 2 — carve out validate set from remaining
        val_ratio = self.val_size / len(remaining)
        strat_rem = remaining[self.strat_col] if strat is not None else None
        train_df, val_df = train_test_split(
            remaining,
            test_size    = val_ratio,
            stratify     = strat_rem,
            random_state = self.random_seed,
        )

        print(f"[Split] Total   : {total}")
        print(f"[Split] Train   : {len(train_df)}")
        print(f"[Split] Validate: {len(val_df)}")
        print(f"[Split] Test    : {len(test_df)}")
        if strat is not None:
            for split_name, split_df in [("train",    train_df),
                                          ("validate", val_df),
                                          ("test",     test_df)]:
                dist = split_df[self.strat_col].value_counts().to_dict()
                print(f"  {split_name} language dist: {dist}")

        return train_df, val_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config_preprocess.yaml")
    config_path = p.parse_args().config
    cfg = load_config(config_path)

    input_path = Path(cfg["input_csv"])
    output_dir = Path(cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Preprocessing] Loading {input_path}")
    df = pd.read_csv(input_path, dtype=str, encoding="utf-8",
                     keep_default_na=False).fillna("")

    # Step 1 — add idx + reorder columns
    print("[Preprocessing] Adding idx column and reordering ...")
    df = reorder_columns(df)
    print(f"[Preprocessing] Column order: {list(df.columns)}")

    # save full file
    full_path = output_dir / "cs-support-multi.csv"
    df.to_csv(full_path, index=False, encoding="utf-8-sig")
    print(f"[Preprocessing] Full file saved → {full_path}")

    # Step 2 — stratified split
    print("[Preprocessing] Splitting ...")
    splitter                  = StratifiedSplitter(cfg)
    train_df, val_df, test_df = splitter.split(df)

    splits = {
        "cs-support-multi-train.csv"   : train_df,
        "cs-support-multi-validate.csv": val_df,
        "cs-support-multi-test.csv"    : test_df,
    }
    for fname, split_df in splits.items():
        split_df = split_df.copy()
        split_df["idx"] = range(len(split_df))
        out_path = output_dir / fname
        split_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[Preprocessing] Saved → {out_path}  ({len(split_df)} rows)")

    print("[Preprocessing] Done.")


if __name__ == "__main__":
    main()
