# data load & clean
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent

TRAIN_PATH = DATA_DIR / "train-test.csv"
VAL_PATH = DATA_DIR / "validation.csv"
DEC_PATH = DATA_DIR / "december-chart-inputs.csv"

REQUIRED_TRAIN_COLS = {
    "load_id", "distance", "equipment", "weight", "date",
    "quote_signal", "posted_rate",
}
REQUIRED_VAL_COLS = {
    "load_id", "distance", "equipment", "weight", "date", "quote_signal",
}


def _impute_weight(df: pd.DataFrame) -> pd.DataFrame:
    # fill nulls using meidan by equipment
    medians = (
        df.groupby("equipment")["weight_clean"]
        .median()
        .to_dict()
    )
    overall_median = df["weight_clean"].median()

    def _fill(row):
        if pd.isna(row["weight_clean"]):
            return medians.get(row["equipment"], overall_median)
        return row["weight_clean"]

    df = df.copy()
    df["weight_clean"] = df.apply(_fill, axis=1)
    return df


def _clean_weight(df: pd.DataFrame) -> pd.DataFrame:
    # handle wight sign inversion from telematics
    df = df.copy()
    df["weight_clean"] = df["weight"].abs()
    df = _impute_weight(df)
    return df


def _parse_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


def _assert_cols(df: pd.DataFrame, required: set, label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_train() -> pd.DataFrame:
    df = pd.read_csv(TRAIN_PATH)
    _assert_cols(df, REQUIRED_TRAIN_COLS, "train-test.csv")
    df = _parse_date(df)
    df = _clean_weight(df)
    assert df["weight_clean"].isna().sum() == 0, "weight imputation incomplete"
    assert (df["weight_clean"] >= 0).all(), "negative weights remain after cleaning"
    return df


def load_validation() -> pd.DataFrame:
    df = pd.read_csv(VAL_PATH)
    _assert_cols(df, REQUIRED_VAL_COLS, "validation.csv")
    df = _parse_date(df)
    df = _clean_weight(df)
    assert df["weight_clean"].isna().sum() == 0, "weight imputation incomplete"
    return df


def load_december() -> pd.DataFrame:
    df = pd.read_csv(DEC_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def build_alpha(df: pd.DataFrame) -> pd.Series:
    # calculate ratio target alpha
    baseline = df["distance"].clip(lower=1.0) * df["quote_signal"]
    return df["posted_rate"] / baseline
