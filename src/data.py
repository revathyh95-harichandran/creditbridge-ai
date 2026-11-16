"""Loading the application data and the one fixed train/test split.

Everything in the modelling part of the project uses the same 80/20 split, made once
and saved to data/processed/split.csv. The test rows are opened exactly once, by
src/evaluate.py, after the final model and cut-off have been fixed on training data.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PROCESSED_DIR, RAW_DIR

SEED = 42           # any fixed number works; used everywhere for reproducibility
TEST_SHARE = 0.20   # 61,503 test applicants, ~5,000 defaulters: enough for a stable final check

APPLICATION_CSV = RAW_DIR / "application_train.csv"
SPLIT_CSV = PROCESSED_DIR / "split.csv"   # git-ignored, like all data

# Never model inputs: protected characteristics (used only for the fairness check).
PROTECTED_COLS = ["CODE_GENDER", "NAME_FAMILY_STATUS"]


def load_application() -> pd.DataFrame:
    """Raw application_train.csv: one row per applicant, TARGET = 1 for payment difficulties."""
    return pd.read_csv(APPLICATION_CSV)


def make_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split once, stratified on TARGET so both parts keep the 8.07% default rate."""
    train_ids, test_ids = train_test_split(
        df["SK_ID_CURR"], test_size=TEST_SHARE, stratify=df["TARGET"], random_state=SEED
    )
    split = pd.concat([
        pd.DataFrame({"SK_ID_CURR": train_ids, "set": "train"}),
        pd.DataFrame({"SK_ID_CURR": test_ids, "set": "test"}),
    ])
    SPLIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_CSV, index=False)
    return split


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) using the saved split; makes the split on first use."""
    df = load_application()
    split = pd.read_csv(SPLIT_CSV) if SPLIT_CSV.exists() else make_split(df)
    df = df.merge(split, on="SK_ID_CURR")
    train = df[df["set"] == "train"].drop(columns="set").reset_index(drop=True)
    test = df[df["set"] == "test"].drop(columns="set").reset_index(drop=True)
    return train, test


def split_xy(df: pd.DataFrame):
    """Inputs (X) and the answer (y)."""
    return df.drop(columns=["TARGET"]), df["TARGET"].to_numpy()
