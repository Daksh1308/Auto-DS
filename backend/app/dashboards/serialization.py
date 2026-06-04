"""DataFrame -> JSON-safe records for the dashboard API."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        ts = pd.Timestamp(value)
        return ts.isoformat() if not pd.isna(ts) else None
    if isinstance(value, pd.Series):
        return [_to_jsonable(v) for v in value.tolist()]
    return value


def dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a list of {column: jsonable_value} records."""
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        out.append({col: _to_jsonable(row[col]) for col in df.columns})
    return out


def dataframe_to_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns]


def sample_dataframe(df: pd.DataFrame, limit: int, seed: int = 42) -> pd.DataFrame:
    """Deterministically sample ``df`` to at most ``limit`` rows."""
    if len(df) <= limit:
        return df
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(df), size=limit, replace=False)
    return df.iloc[sorted(indices.tolist())].reset_index(drop=True)
