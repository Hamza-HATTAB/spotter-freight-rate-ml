# rolling origin temporal splits
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.data import build_alpha, load_train
from src.features import engineer, get_feature_matrix

ALPHA_CLIP = (0.5, 3.5)

SPLITS = [
    ("2025-01-01", "2025-06-30", "2025-07-01", "2025-08-31", "Jul-Aug"),
    ("2025-01-01", "2025-08-31", "2025-09-01", "2025-10-31", "Sep-Oct"),
]


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _medape(y_true, y_pred):
    # meidan absolute percentage error
    return float(np.median(np.abs((y_true - y_pred) / y_true))) * 100


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_validation() -> list[dict]:
    df = load_train()
    df = engineer(df)
    df["alpha"] = build_alpha(df)

    results = []
    for train_start, train_end, test_start, test_end, label in SPLITS:
        mask_tr = (df["date"] >= train_start) & (df["date"] <= train_end)
        mask_te = (df["date"] >= test_start) & (df["date"] <= test_end)

        tr = df[mask_tr]
        te = df[mask_te]

        X_tr = get_feature_matrix(tr)
        y_tr = tr["alpha"].values

        X_te = get_feature_matrix(te)
        y_te_rate = te["posted_rate"].values
        baseline_te = te["distance"].clip(lower=1.0) * te["quote_signal"]

        # L1 loss minimises conditional median
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=300,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        alpha_hat = model.predict(X_te)
        alpha_hat = np.clip(alpha_hat, *ALPHA_CLIP)
        rate_hat = baseline_te.values * alpha_hat

        row = {
            "fold": label,
            "train_rows": len(tr),
            "test_rows": len(te),
            "mae": _mae(y_te_rate, rate_hat),
            "medape_pct": _medape(y_te_rate, rate_hat),
            "rmse": _rmse(y_te_rate, rate_hat),
        }
        results.append(row)
        print(
            f"  [{label}] train={len(tr):,} test={len(te):,} "
            f"MAE=${row['mae']:.1f} MedAPE={row['medape_pct']:.2f}% RMSE=${row['rmse']:.1f}"
        )

    return results


if __name__ == "__main__":
    print("Running rolling-origin validation...")
    run_validation()
