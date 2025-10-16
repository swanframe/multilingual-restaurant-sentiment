# src/data/datasets.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import load_config
from src.utils.io import ensure_dir
from src.data.preprocess import basic_clean_multilingual

REQUIRED_COLS = {"text", "language", "sentiment"}
ALLOWED_LANGS = {"en", "id"}
ALLOWED_LABELS = {"positive", "negative", "neutral"}

def _read_csv_safe(path: Path) -> pd.DataFrame:
    # No pyarrow; use default engine
    return pd.read_csv(path, encoding="utf-8")

def load_ai_data(cfg: Dict) -> pd.DataFrame:
    ai_glob = cfg["data"]["ai_glob"]
    paths = [Path(p) for p in sorted(Path().glob(ai_glob))]
    if not paths:
        raise FileNotFoundError(f"No AI CSVs found for pattern: {ai_glob}")
    frames = []
    for p in paths:
        df = _read_csv_safe(p)
        df["__source_file"] = str(p)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def validate_schema(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. "
                         f"Expected columns: {sorted(REQUIRED_COLS)}")
    extra = set(df.columns) - (REQUIRED_COLS | {"number", "restaurant_type", "__source_file"})
    # extra columns are allowed; we just won't use them.

def validate_values(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with nulls in required columns
    df = df.dropna(subset=list(REQUIRED_COLS)).copy()

    # Strip and normalize critical fields
    df["text"] = df["text"].astype(str)
    df["language"] = df["language"].astype(str).str.strip().str.lower()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()

    # Filter to allowed sets
    before = len(df)
    df = df[df["language"].isin(ALLOWED_LANGS)]
    df = df[df["sentiment"].isin(ALLOWED_LABELS)]
    after = len(df)
    if after < before:
        print(f"[validate_values] Dropped {before - after} rows due to invalid language/sentiment.")
    return df

def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_clean"] = df["text"].apply(basic_clean_multilingual)
    # Drop rows that become empty after cleaning
    before = len(df)
    df = df[df["text_clean"].str.len() > 0]
    dropped = before - len(df)
    if dropped:
        print(f"[clean_text] Dropped {dropped} rows with empty text after cleaning.")
    return df

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # De-dup by (text_clean, language, sentiment)
    before = len(df)
    df = df.drop_duplicates(subset=["text_clean", "language", "sentiment"]).reset_index(drop=True)
    deduped = before - len(df)
    if deduped:
        print(f"[drop_duplicates] Removed {deduped} duplicate rows.")
    return df

def summarize(df: pd.DataFrame, title: str = "Dataset Summary") -> None:
    print(f"\n=== {title} ===")
    print(f"Rows: {len(df)}")
    print("By language:\n", df["language"].value_counts(dropna=False))
    print("By sentiment:\n", df["sentiment"].value_counts(dropna=False))
    print("By language+sentiment:\n", df.groupby(["language", "sentiment"]).size())

def stratified_splits(
    df: pd.DataFrame, cfg: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratios = cfg["splits"]
    train_ratio, val_ratio, test_ratio = ratios["train"], ratios["val"], ratios["test"]
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Splits must sum to 1.0")

    # We stratify on combined key to preserve both language and sentiment
    df = df.copy()
    df["stratify_key"] = df["language"] + "__" + df["sentiment"]

    # First: train vs temp (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - train_ratio), random_state=cfg["train"]["seed"]
    )
    idx_train, idx_temp = next(sss1.split(df, df["stratify_key"]))
    train_df = df.iloc[idx_train].reset_index(drop=True)
    temp_df = df.iloc[idx_temp].reset_index(drop=True)

    # Second: split temp into val and test with preserved ratio
    val_prop = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1.0 - val_prop), random_state=cfg["train"]["seed"]
    )
    idx_val, idx_test = next(sss2.split(temp_df, temp_df["stratify_key"]))
    val_df = temp_df.iloc[idx_val].reset_index(drop=True)
    test_df = temp_df.iloc[idx_test].reset_index(drop=True)

    # Safety: do not include real data here (we never loaded it — but double-check paths if present)
    for part, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        if "__source_file" in part.columns and part["__source_file"].astype(str).str.contains("/data/real/").any():
            raise RuntimeError(f"Real data leaked into {name} split. Check your file locations.")

    return train_df, val_df, test_df

def save_outputs(ai_clean: pd.DataFrame, splits: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], cfg: Dict) -> None:
    out_root = Path(cfg["paths"]["outputs_dir"])
    data_dir = out_root / "data"
    splits_dir = out_root / "splits"
    ensure_dir(data_dir)
    ensure_dir(splits_dir)

    ai_path = data_dir / "ai_combined_clean.csv"
    ai_clean.to_csv(ai_path, index=False, encoding="utf-8")

    train_df, val_df, test_df = splits
    train_df.to_csv(splits_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(splits_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(splits_dir / "test.csv", index=False, encoding="utf-8")

    print(f"\nSaved:")
    print(f"  - {ai_path}")
    print(f"  - {splits_dir / 'train.csv'}")
    print(f"  - {splits_dir / 'val.csv'}")
    print(f"  - {splits_dir / 'test.csv'}")

def build_clean_splits() -> None:
    cfg = load_config()

    # 1) Load + validate
    df = load_ai_data(cfg)
    validate_schema(df)
    df = validate_values(df)

    # 2) Clean + dedup
    df = clean_text(df)
    df = drop_duplicates(df)

    # 3) Summaries
    summarize(df, "AI Combined (Cleaned)")

    # 4) Stratified splits (by language+sentiment)
    train_df, val_df, test_df = stratified_splits(df, cfg)

    summarize(train_df, "Train")
    summarize(val_df, "Validation")
    summarize(test_df, "Test")

    # 5) Save
    save_outputs(df, (train_df, val_df, test_df), cfg)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="Print dataset summary after processing.")
    args = parser.parse_args()

    build_clean_splits()

    if args.summary:
        # Already printed inside build_clean_splits()
        pass

if __name__ == "__main__":
    main()