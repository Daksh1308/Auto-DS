"""Top-vs-bottom category comparison for each (categorical, numeric) pair."""

from __future__ import annotations

import math

import pandas as pd


_CATEGORICAL_TYPES = {"categorical", "boolean"}


def _safe_float(value: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compute_category_breakdowns(
    df: pd.DataFrame, types: dict[str, str]
) -> list[dict]:
    """Return one record per (categorical, numeric) pair, ranked by |%diff|."""
    cat_cols = [c for c, t in types.items() if t in _CATEGORICAL_TYPES]
    num_cols = [c for c, t in types.items() if t in {"integer", "float"}]
    if not cat_cols or not num_cols:
        return []

    results: list[dict] = []
    for cat_col in cat_cols:
        for num_col in num_cols:
            numeric = pd.to_numeric(df[num_col], errors="coerce")
            grouped = pd.DataFrame(
                {"cat": df[cat_col], "num": numeric}
            ).dropna(subset=["cat", "num"])
            if grouped.empty:
                continue
            means = grouped.groupby("cat", observed=True)["num"].mean()
            if len(means) < 2:
                continue
            ordered = means.sort_values(ascending=False)
            top_label = str(ordered.index[0])
            top_mean = _safe_float(ordered.iloc[0])
            bottom_label = str(ordered.index[-1])
            bottom_mean = _safe_float(ordered.iloc[-1])
            if top_mean is None or bottom_mean is None or bottom_mean == 0:
                continue
            pct_diff = (top_mean - bottom_mean) / abs(bottom_mean) * 100.0
            results.append(
                {
                    "category_col": cat_col,
                    "numeric_col": num_col,
                    "top": {"value": top_label, "mean": round(top_mean, 4)},
                    "bottom": {"value": bottom_label, "mean": round(bottom_mean, 4)},
                    "pct_diff": round(pct_diff, 2),
                }
            )

    results.sort(key=lambda r: abs(r["pct_diff"]), reverse=True)
    return results
