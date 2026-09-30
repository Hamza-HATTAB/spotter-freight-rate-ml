"""
Feature engineering for the freight rate model.

Retained features: distance (raw, log, inverse), quote_signal, is_short_haul,
weight_clean, day_of_week, equipment dummies.

Dropped: market_index (regime drift), lat/lon (subsumed by distance),
day_of_year harmonics (Jan-Oct cannot extrapolate to Nov-Dec phase).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# threshold determined by cross-validated MAE on rolling splits (tested 150, 300, 450, 600)
SHORT_HAUL_THRESHOLD = 450.0

FEATURE_COLS = [
    "distance",
    "log_distance",
    "inv_distance",
    "quote_signal",
    "is_short_haul",
    "weight_clean",
    "day_of_week",
    "equipment_Reefer",
    "equipment_Flatbed",
]


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with all model features added as columns."""
    out = df.copy()

    out["log_distance"] = np.log1p(out["distance"])
    # calulate inverse distance for non-linear rate-per-mile decay
    out["inv_distance"] = 1.0 / out["distance"].clip(lower=1.0)
    out["is_short_haul"] = (out["distance"] < SHORT_HAUL_THRESHOLD).astype(np.int8)
    out["day_of_week"] = out["date"].dt.dayofweek
    out["equipment_Reefer"] = (out["equipment"] == "Reefer").astype(np.int8)
    out["equipment_Flatbed"] = (out["equipment"] == "Flatbed").astype(np.int8)

    return out


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the model input matrix from an engineered dataframe."""
    return df[FEATURE_COLS].copy()
