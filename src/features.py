# 9 core tabular features
from __future__ import annotations

import numpy as np
import pandas as pd

# short haul cuttoff from validation
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
    out = df.copy()

    out["log_distance"] = np.log1p(out["distance"])
    # calulate inverse distance for non-linear rate decay
    out["inv_distance"] = 1.0 / out["distance"].clip(lower=1.0)
    out["is_short_haul"] = (out["distance"] < SHORT_HAUL_THRESHOLD).astype(np.int8)
    out["day_of_week"] = out["date"].dt.dayofweek
    out["equipment_Reefer"] = (out["equipment"] == "Reefer").astype(np.int8)
    out["equipment_Flatbed"] = (out["equipment"] == "Flatbed").astype(np.int8)

    return out


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLS].copy()
