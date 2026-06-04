"""Auto-suggest chart specs for a cleaned DataFrame."""

from __future__ import annotations

import math
import uuid
from typing import Literal

import numpy as np
import pandas as pd

from ..cleaner.type_detect import infer_column_types
from .plotly_specs import build_plotly_spec


ChartType = Literal["line", "bar", "pie"]
Aggregation = Literal["sum", "mean", "count", "median", "min", "max"]


def _safe_float(value: float) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _score_chart(values: pd.Series) -> float:
    """Higher = more interesting.

    Uses coefficient of variation (std / mean) on the aggregated y values,
    which rewards variety without rewarding raw count imbalance, plus a
    small bonus for having more unique x values.
    """
    non_null = values.dropna()
    if non_null.empty:
        return 0.0
    vmean = float(non_null.mean())
    if vmean == 0 or len(non_null) < 2:
        return 0.0
    vstd = float(non_null.std())
    cv = vstd / abs(vmean)
    n_unique = int(non_null.nunique())
    variety_bonus = math.log2(max(n_unique, 2)) * 0.5
    return cv + variety_bonus


def _make_spec(
    chart_type: ChartType,
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    aggregation: Aggregation,
    title: str,
) -> dict:
    plotly_spec = build_plotly_spec(
        chart_type=chart_type,
        df=df,
        x_column=x_column,
        y_column=y_column,
        aggregation=aggregation,
        title=title,
    )
    return {
        "id": uuid.uuid4().hex,
        "type": chart_type,
        "title": title,
        "x_column": x_column,
        "y_column": y_column,
        "aggregation": aggregation,
        "plotly_spec": plotly_spec,
    }


def _score_spec_against_dataframe(
    df: pd.DataFrame, spec: dict
) -> float:
    """Recompute the score for a generated spec by aggregating its data."""
    from .plotly_specs import aggregate

    try:
        agg = aggregate(df, spec["x_column"], spec.get("y_column"), spec["aggregation"])
    except ValueError:
        return 0.0
    if agg.empty or "y" not in agg.columns:
        return 0.0
    return _score_chart(agg["y"])


def suggest_charts(
    df: pd.DataFrame,
    types: dict[str, str] | None = None,
    max_charts: int = 8,
) -> list[dict]:
    """Return up to ``max_charts`` ChartSpec dicts ranked by interestingness.

    Rules:
        - one line per (datetime, numeric) pair, aggregation=sum
        - one bar per (categorical, numeric) pair, aggregation=sum
        - one count-bar per categorical with no numeric partner
        - one pie per categorical column, value=count
    """
    if types is None:
        types = infer_column_types(df)

    datetime_cols = [c for c, t in types.items() if t == "datetime"]
    numeric_cols = [c for c, t in types.items() if t in {"integer", "float"}]
    categorical_cols = [
        c for c, t in types.items() if t in {"categorical", "boolean"}
    ]

    candidates: list[tuple[float, dict]] = []

    # Lines: (datetime, numeric)
    for dt_col in datetime_cols:
        for num_col in numeric_cols:
            if dt_col == num_col:
                continue
            try:
                spec = _make_spec(
                    "line",
                    df,
                    dt_col,
                    num_col,
                    "sum",
                    f"{num_col} over time",
                )
            except ValueError:
                continue
            score = _score_spec_against_dataframe(df, spec)
            candidates.append((score, spec))

    # Bars: (categorical, numeric) — use mean so the chart is "avg by category"
    # rather than count-dominated totals.
    paired_cat: set[str] = set()
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            try:
                spec = _make_spec(
                    "bar",
                    df,
                    cat_col,
                    num_col,
                    "mean",
                    f"avg {num_col} by {cat_col}",
                )
            except ValueError:
                continue
            score = _score_spec_against_dataframe(df, spec)
            candidates.append((score, spec))
            paired_cat.add(cat_col)

    # Count bars for un-paired categoricals
    for cat_col in categorical_cols:
        if cat_col in paired_cat:
            continue
        try:
            spec = _make_spec(
                "bar", df, cat_col, None, "count", f"Count by {cat_col}"
            )
        except ValueError:
            continue
        score = _score_spec_against_dataframe(df, spec)
        candidates.append((score, spec))

    # Pies: one per categorical
    for cat_col in categorical_cols:
        try:
            spec = _make_spec(
                "pie", df, cat_col, None, "count", f"{cat_col} distribution"
            )
        except ValueError:
            continue
        score = _score_spec_against_dataframe(df, spec)
        candidates.append((score, spec))

    # Sort by score desc, then by stable order (id tiebreak), and trim.
    candidates.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [spec for _, spec in candidates[:max_charts]]


def column_types_for(df: pd.DataFrame) -> dict[str, str]:
    """Helper exposed for the API: infer and return column types for the dashboard."""
    return infer_column_types(df)
