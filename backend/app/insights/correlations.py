"""Pairwise Pearson correlations for numeric columns."""

from __future__ import annotations

import math

import pandas as pd


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


def compute_correlations(
    df: pd.DataFrame, numeric_cols: list[str], min_abs_r: float = 0.5
) -> list[dict]:
    """Return a sorted, dedup'd list of strong correlations between numeric columns.

    Each entry is ``{a, b, r}``. Pairs are returned once (lexicographic) and
    sorted by ``|r|`` descending.
    """
    if len(numeric_cols) < 2:
        return []

    numeric_df = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr()

    seen: set[tuple[str, str]] = set()
    results: list[dict] = []
    for a in corr.columns:
        for b in corr.columns:
            if a == b:
                continue
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            r = _safe_float(corr.loc[a, b])
            if r is None:
                continue
            if abs(r) >= min_abs_r:
                results.append({"a": key[0], "b": key[1], "r": round(r, 4)})

    results.sort(key=lambda x: abs(x["r"]), reverse=True)
    return results
