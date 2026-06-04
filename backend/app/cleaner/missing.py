"""Missing-value handling for the cleaning pipeline."""

from __future__ import annotations

import pandas as pd


HIGH_NULL_DROP_THRESHOLD = 0.6


def _fill_numeric(series: pd.Series) -> tuple[pd.Series, str]:
    if series.notna().any():
        fill_value = series.median()
        return series.fillna(fill_value), f"filled with median ({fill_value})"
    return series, "left empty (no non-null values to compute median)"


def _fill_categorical(series: pd.Series) -> tuple[pd.Series, str]:
    if series.notna().any():
        mode = series.mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "unknown"
        return series.fillna(fill_value), f"filled with mode ('{fill_value}')"
    return series.fillna("unknown"), "filled with constant 'unknown'"


def _fill_text(series: pd.Series) -> tuple[pd.Series, str]:
    return series.fillna("unknown"), "filled with constant 'unknown'"


def _fill_datetime(series: pd.Series) -> tuple[pd.Series, str]:
    coerced = pd.to_datetime(series, errors="coerce", format="mixed")
    filled = coerced.ffill()
    if filled.isna().any():
        filled = filled.bfill()
    return filled, "forward-filled (with backfill fallback)"


_FILLERS = {
    "integer": _fill_numeric,
    "float": _fill_numeric,
    "categorical": _fill_categorical,
    "text": _fill_text,
    "boolean": _fill_categorical,
    "datetime": _fill_datetime,
}


def fill_missing(
    df: pd.DataFrame, types: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    """Fill missing values per the inferred type map.

    Returns the filled frame, an action map (column -> description of what was
    done), and the list of columns dropped because too many values were null.
    """
    out = df.copy()
    actions: dict[str, str] = {}
    dropped: list[str] = []

    for col in list(out.columns):
        null_ratio = out[col].isna().mean() if len(out) else 0.0
        if null_ratio > HIGH_NULL_DROP_THRESHOLD:
            dropped.append(col)
            out = out.drop(columns=[col])
            actions[col] = f"dropped ({null_ratio:.0%} null)"
            continue
        col_type = types.get(col, "text")
        filler = _FILLERS.get(col_type, _fill_text)
        out[col], action = filler(out[col])
        actions[col] = action

    return out, actions, dropped
