"""Build Plotly figure dicts from a ChartSpec + DataFrame."""

from __future__ import annotations

import math
from typing import Any, Literal

import pandas as pd


ChartType = Literal["line", "bar", "pie"]
Aggregation = Literal["sum", "mean", "count", "median", "min", "max"]


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _jsonify_x(values: pd.Series) -> list:
    """Convert a Series of x-values to JSON-safe primitives (str for datetimes)."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return [v.isoformat() if pd.notna(v) else None for v in values]
    return [None if pd.isna(v) else v for v in values.tolist()]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def aggregate(
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    aggregation: Aggregation,
) -> pd.DataFrame:
    """Group ``df`` by ``x_column`` and aggregate ``y_column``.

    When ``y_column`` is None or ``aggregation == 'count'``, count rows per x.
    Returns a DataFrame with columns ``[x_column, y]``.
    """
    if x_column not in df.columns:
        raise ValueError(f"x_column '{x_column}' not in DataFrame")

    if y_column is None or aggregation == "count":
        grouped = df.groupby(x_column, dropna=True).size().reset_index(name="y")
        return grouped

    if y_column not in df.columns:
        raise ValueError(f"y_column '{y_column}' not in DataFrame")

    numeric = _coerce_numeric(df[y_column])
    if aggregation == "sum":
        agg = numeric.groupby(df[x_column], dropna=True).sum()
    elif aggregation == "mean":
        agg = numeric.groupby(df[x_column], dropna=True).mean()
    elif aggregation == "median":
        agg = numeric.groupby(df[x_column], dropna=True).median()
    elif aggregation == "min":
        agg = numeric.groupby(df[x_column], dropna=True).min()
    elif aggregation == "max":
        agg = numeric.groupby(df[x_column], dropna=True).max()
    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'")

    return agg.reset_index().rename(columns={y_column: "y"})


def _sort_for_x(df: pd.DataFrame, x_column: str) -> pd.DataFrame:
    out = df.copy()
    x = out[x_column]
    if pd.api.types.is_datetime64_any_dtype(x):
        out = out.sort_values(x_column)
    elif pd.api.types.is_numeric_dtype(x):
        out = out.sort_values(x_column)
    else:
        # Categorical/text: sort by y desc so the most prominent category comes first.
        out = out.sort_values("y", ascending=False)
    return out.reset_index(drop=True)


def _build_line(df: pd.DataFrame, x_column: str, y_column: str | None, title: str) -> dict:
    if y_column is None:
        raise ValueError("line chart requires a y_column")
    sorted_df = _sort_for_x(df, x_column)
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines+markers",
                "x": _jsonify_x(sorted_df[x_column]),
                "y": [_safe_float(v) for v in sorted_df["y"].tolist()],
                "name": y_column,
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": x_column}},
            "yaxis": {"title": {"text": y_column}},
            "margin": {"l": 50, "r": 20, "t": 50, "b": 50},
        },
    }


def _build_bar(df: pd.DataFrame, x_column: str, y_column: str | None, title: str) -> dict:
    sorted_df = _sort_for_x(df, x_column)
    return {
        "data": [
            {
                "type": "bar",
                "x": _jsonify_x(sorted_df[x_column]),
                "y": [_safe_float(v) for v in sorted_df["y"].tolist()],
                "name": y_column or "count",
            }
        ],
        "layout": {
            "title": {"text": title},
            "xaxis": {"title": {"text": x_column}},
            "yaxis": {"title": {"text": y_column or "count"}},
            "margin": {"l": 50, "r": 20, "t": 50, "b": 80},
        },
    }


def _build_pie(df: pd.DataFrame, x_column: str, y_column: str | None, title: str) -> dict:
    sorted_df = _sort_for_x(df, x_column)
    return {
        "data": [
            {
                "type": "pie",
                "labels": _jsonify_x(sorted_df[x_column]),
                "values": [_safe_float(v) for v in sorted_df["y"].tolist()],
                "name": x_column,
            }
        ],
        "layout": {
            "title": {"text": title},
            "margin": {"l": 20, "r": 20, "t": 50, "b": 20},
        },
    }


_BUILDERS = {
    "line": _build_line,
    "bar": _build_bar,
    "pie": _build_pie,
}


def build_plotly_spec(
    chart_type: ChartType,
    df: pd.DataFrame,
    x_column: str,
    y_column: str | None,
    aggregation: Aggregation,
    title: str,
) -> dict:
    """Aggregate ``df`` per the spec and return a Plotly figure dict."""
    if chart_type not in _BUILDERS:
        raise ValueError(f"Unknown chart type '{chart_type}'")
    aggregated = aggregate(df, x_column, y_column, aggregation)
    if aggregated.empty:
        # Plotly doesn't handle empty data well; return a placeholder trace so the
        # frontend still has a valid figure object to render.
        return {
            "data": [],
            "layout": {
                "title": {"text": f"{title} (no data)"},
                "annotations": [
                    {
                        "text": "No data for this combination",
                        "showarrow": False,
                        "xref": "paper",
                        "yref": "paper",
                        "x": 0.5,
                        "y": 0.5,
                    }
                ],
            },
        }
    return _BUILDERS[chart_type](aggregated, x_column, y_column, title)
