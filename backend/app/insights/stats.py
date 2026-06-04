"""Per-column summary statistics for the insights pipeline.

Output shape per column is a dict whose keys depend on the column's inferred
type. Keeping it as a plain dict (not a pydantic model) lets the API layer
serialize a single open shape and lets tests assert with ``in``.
"""

from __future__ import annotations

import math

import pandas as pd

from .outliers import iqr_outlier_count


_NUMERIC_TYPES = {"integer", "float"}
_CATEGORICAL_TYPES = {"categorical", "boolean"}


def _safe_float(value: float) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _numeric_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    if non_null.empty:
        return {"count": 0}
    return {
        "count": int(len(non_null)),
        "mean": _safe_float(non_null.mean()),
        "median": _safe_float(non_null.median()),
        "std": _safe_float(non_null.std()),
        "min": _safe_float(non_null.min()),
        "max": _safe_float(non_null.max()),
        "q25": _safe_float(non_null.quantile(0.25)),
        "q75": _safe_float(non_null.quantile(0.75)),
        "skew": _safe_float(non_null.skew()),
        "outliers_iqr": iqr_outlier_count(series),
    }


def _categorical_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    if non_null.empty:
        return {"count": 0, "unique": 0}
    counts = non_null.value_counts()
    top_value = counts.index[0]
    top_count = int(counts.iloc[0])
    return {
        "count": int(len(non_null)),
        "unique": int(non_null.nunique()),
        "top": str(top_value),
        "top_count": top_count,
        "top_pct": _safe_float(top_count / len(non_null) * 100.0),
    }


def _text_stats(series: pd.Series) -> dict:
    non_null = series.dropna()
    if non_null.empty:
        return {"count": 0, "unique": 0}
    return {
        "count": int(len(non_null)),
        "unique": int(non_null.nunique()),
        "avg_length": _safe_float(non_null.astype(str).str.len().mean()),
        "max_length": _safe_float(non_null.astype(str).str.len().max()),
    }


def _datetime_stats(series: pd.Series) -> dict:
    coerced = pd.to_datetime(series, errors="coerce", format="mixed")
    non_null = coerced.dropna()
    if non_null.empty:
        return {"count": 0}
    span = non_null.max() - non_null.min()
    return {
        "count": int(len(non_null)),
        "min": non_null.min().isoformat(),
        "max": non_null.max().isoformat(),
        "span_days": int(span.days),
    }


def compute_column_stats(
    df: pd.DataFrame, types: dict[str, str]
) -> dict[str, dict]:
    """Return a mapping of column name -> stats dict."""
    out: dict[str, dict] = {}
    for col in df.columns:
        col_type = types.get(col, "text")
        if col_type in _NUMERIC_TYPES:
            stats = _numeric_stats(df[col])
        elif col_type in _CATEGORICAL_TYPES:
            stats = _categorical_stats(df[col])
        elif col_type == "datetime":
            stats = _datetime_stats(df[col])
        else:
            stats = _text_stats(df[col])
        out[col] = {"type": col_type, "stats": stats}
    return out
