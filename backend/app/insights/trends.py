"""Numeric-over-datetime trend analysis.

Picks the primary datetime column (widest range), bins the time axis into 4
equal-width periods (quarters / months / weeks chosen by span), and for every
numeric column reports the period-over-period % change from the first to the
last non-empty period.
"""

from __future__ import annotations

import math

import pandas as pd


_QUARTER_DAYS = 365
_MONTH_DAYS = 90


def _pick_kind(span_days: int) -> str:
    if span_days >= _QUARTER_DAYS:
        return "quarter"
    if span_days >= _MONTH_DAYS:
        return "month"
    return "week"


def _label_for_date(d: pd.Timestamp, kind: str) -> str:
    if kind == "quarter":
        q = (d.month - 1) // 3 + 1
        return f"{d.year}-Q{q}"
    if kind == "month":
        return f"{d.year}-{d.month:02d}"
    iso = d.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def _bin_to_4_periods(
    dt_series: pd.Series,
) -> tuple[pd.Series, list[str], str] | tuple[None, None, None]:
    non_null = dt_series.dropna()
    if non_null.empty:
        return None, None, None
    min_d = non_null.min()
    max_d = non_null.max()
    span_days = int((max_d - min_d).days)
    if span_days <= 0:
        return None, None, None

    kind = _pick_kind(span_days)
    edges = pd.date_range(start=min_d, end=max_d, periods=5)
    labels = [_label_for_date(edges[i], kind) for i in range(4)]
    binned = pd.cut(dt_series, bins=edges, labels=labels, include_lowest=True)
    return binned, labels, kind


def _safe_float(value: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _summarize_period(
    numeric: pd.Series, periods: pd.Series
) -> tuple[float, float, str, str] | None:
    """Sum numeric within first and last non-empty periods.

    Returns ``(first_value, last_value, first_label, last_label)`` or ``None``
    if there are not enough populated periods to compare.
    """
    valid = numeric.notna() & periods.notna()
    if valid.sum() == 0:
        return None
    grouped = pd.DataFrame({"n": numeric, "p": periods}).dropna()
    sums = grouped.groupby("p", observed=True)["n"].sum()
    non_empty = sums[sums != 0]
    if len(non_empty) < 2:
        return None
    ordered = sums.index.tolist()
    first_label = ordered[0]
    last_label = ordered[-1]
    first_value = float(sums.iloc[0])
    last_value = float(sums.iloc[-1])
    if first_value == 0:
        return None
    return first_value, last_value, str(first_label), str(last_label)


def compute_trends(
    df: pd.DataFrame, types: dict[str, str]
) -> list[dict]:
    """Return one trend record per (primary datetime, numeric) pair."""
    datetime_cols = [c for c, t in types.items() if t == "datetime"]
    if not datetime_cols:
        return []

    # Pick the datetime column with the widest span (most informative trend).
    best_col: str | None = None
    best_span = -1
    for col in datetime_cols:
        coerced = pd.to_datetime(df[col], errors="coerce", format="mixed")
        non_null = coerced.dropna()
        if non_null.empty:
            continue
        span = int((non_null.max() - non_null.min()).days)
        if span > best_span:
            best_span = span
            best_col = col
    if best_col is None:
        return []

    coerced_dt = pd.to_datetime(df[best_col], errors="coerce", format="mixed")
    binned, _labels, _kind = _bin_to_4_periods(coerced_dt)
    if binned is None:
        return []

    numeric_cols = [
        c for c, t in types.items() if t in {"integer", "float"} and c != best_col
    ]

    results: list[dict] = []
    for col in numeric_cols:
        summary = _summarize_period(
            pd.to_numeric(df[col], errors="coerce"), binned
        )
        if summary is None:
            continue
        first_value, last_value, first_label, last_label = summary
        pct = (last_value - first_value) / abs(first_value) * 100.0
        direction = "up" if pct >= 0 else "down"
        results.append(
            {
                "column": col,
                "period_col": best_col,
                "first_period": first_label,
                "last_period": last_label,
                "first_value": round(first_value, 4),
                "last_value": round(last_value, 4),
                "pct_change": round(pct, 2),
                "direction": direction,
            }
        )

    # Sort by |pct_change| descending so the most newsworthy trends surface first.
    results.sort(key=lambda r: abs(r["pct_change"]), reverse=True)
    return results
