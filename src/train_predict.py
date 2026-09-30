# train on full 48k loads and predict holdout
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import build_alpha, load_december, load_train, load_validation
from src.features import engineer, get_feature_matrix

ALPHA_CLIP = (0.5, 3.5)
OUT_DIR = Path(__file__).parent.parent
PRED_PATH = OUT_DIR / "validation_predictions.csv"
DEC_PATH = OUT_DIR / "december-chart-inputs.csv"


def _build_model() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=300,
        random_state=42,
    )


def _predict_rates(
    model: HistGradientBoostingRegressor,
    df: pd.DataFrame,
) -> np.ndarray:
    X = get_feature_matrix(df)
    alpha_hat = model.predict(X)
    alpha_hat = np.clip(alpha_hat, *ALPHA_CLIP)
    # calculate baseline distance * quote_singal
    baseline = df["distance"].clip(lower=1.0) * df["quote_signal"]
    rates = baseline.values * alpha_hat
    # gaurantee positive rates after clip
    return np.maximum(rates, 1.0)


def _compute_december_quote_signal(val: pd.DataFrame) -> dict:
    # 7-day rolling mean on daily quote_signal
    daily = (
        val.groupby("date")["quote_signal"]
        .mean()
        .sort_index()
    )
    smoothed = daily.rolling(window=7, min_periods=1).mean()
    return smoothed.to_dict()


def run() -> None:
    print("Loading and cleaning training data...")
    train = load_train()
    train = engineer(train)
    train["alpha"] = build_alpha(train)

    print(f"Training HistGBT on {len(train):,} rows...")
    model = _build_model()
    X_train = get_feature_matrix(train)
    model.fit(X_train, train["alpha"].values)

    print("Loading validation set and generating predictions...")
    val = load_validation()
    val = engineer(val)
    val["predicted_rate"] = _predict_rates(model, val)

    preds = val[["load_id", "predicted_rate"]].copy()
    assert len(preds) == 12_000, f"expected 12000 rows, got {len(preds)}"
    assert (preds["predicted_rate"] > 0).all(), "non-positive rates found"
    preds.to_csv(PRED_PATH, index=False)
    print(f"Saved {len(preds):,} predictions to {PRED_PATH}")

    print("Computing December quote_signal signal...")
    val_raw = load_validation()
    val_raw["date"] = pd.to_datetime(val_raw["date"])
    qs_map = _compute_december_quote_signal(val_raw)

    dec = load_december()
    dec["quote_signal"] = dec["date"].map(qs_map)

    # fallback: use overall val quote_signal mean if any December dates missing
    fallback_qs = val_raw["quote_signal"].mean()
    missing_qs = dec["quote_signal"].isna().sum()
    if missing_qs > 0:
        print(f"  {missing_qs} December dates missing quote_signal — using fallback {fallback_qs:.4f}")
        dec["quote_signal"] = dec["quote_signal"].fillna(fallback_qs)

    dec["weight_clean"] = dec["weight"].abs()

    dec = engineer(dec)
    dec["predicted_rate"] = _predict_rates(model, dec)

    # write predictions back into the original december file (preserving all columns)
    dec_out = load_december()
    dec_out["predicted_rate"] = dec["predicted_rate"].values
    assert len(dec_out) == 31, "December output must have exactly 31 rows"
    assert (dec_out["predicted_rate"] > 0).all(), "non-positive December rates"
    dec_out.to_csv(DEC_PATH, index=False)
    print(f"Saved 31 December predictions to {DEC_PATH}")

    rate_summary = val["predicted_rate"]
    print(
        f"\nValidation rate summary — "
        f"min=${rate_summary.min():.0f} "
        f"median=${rate_summary.median():.0f} "
        f"max=${rate_summary.max():.0f}"
    )
    dec_summary = dec_out["predicted_rate"]
    print(
        f"December rate summary — "
        f"min=${dec_summary.min():.0f} "
        f"median=${dec_summary.median():.0f} "
        f"max=${dec_summary.max():.0f}"
    )


if __name__ == "__main__":
    run()
