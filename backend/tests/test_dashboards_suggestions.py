"""Tests for app.dashboards.suggestions."""

from __future__ import annotations

import pandas as pd

from app.dashboards.suggestions import suggest_charts


def test_suggest_lines_for_datetime_numeric_pairs() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "sales": [10, 20, 30, 40],
            "units": [1, 2, 3, 4],
        }
    )
    types = {"d": "datetime", "sales": "integer", "units": "integer"}
    specs = suggest_charts(df, types=types, max_charts=10)
    lines = [s for s in specs if s["type"] == "line"]
    assert len(lines) == 2
    cols = {s["x_column"] for s in lines}
    assert cols == {"d"}


def test_suggest_bars_for_categorical_numeric_pairs() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "S", "N", "S"] * 3,
            "sales": [10, 20, 30, 40] * 3,
        }
    )
    types = {"region": "categorical", "sales": "integer"}
    specs = suggest_charts(df, types=types, max_charts=10)
    bars = [s for s in specs if s["type"] == "bar"]
    assert len(bars) == 1
    assert bars[0]["x_column"] == "region"
    assert bars[0]["y_column"] == "sales"
    assert bars[0]["aggregation"] == "mean"


def test_suggest_count_bar_for_lone_categorical() -> None:
    df = pd.DataFrame({"region": ["N", "S", "N", "E", "N"]})
    types = {"region": "categorical"}
    specs = suggest_charts(df, types=types, max_charts=10)
    bars = [s for s in specs if s["type"] == "bar"]
    assert len(bars) == 1
    assert bars[0]["y_column"] is None
    assert bars[0]["aggregation"] == "count"


def test_suggest_pie_for_each_categorical() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "S", "E", "W"] * 5,
            "sub": ["yes", "no"] * 10,
        }
    )
    types = {"region": "categorical", "sub": "boolean"}
    specs = suggest_charts(df, types=types, max_charts=10)
    pies = [s for s in specs if s["type"] == "pie"]
    assert len(pies) == 2


def test_suggest_caps_at_max_charts() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-02"] * 10),
            "a": [1, 2] * 10,
            "b": [3, 4] * 10,
            "c": ["x", "y"] * 10,
        }
    )
    types = {"d": "datetime", "a": "integer", "b": "integer", "c": "categorical"}
    specs = suggest_charts(df, types=types, max_charts=3)
    assert len(specs) == 3


def test_suggest_returns_empty_for_empty_types() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    types = {"a": "integer"}
    assert suggest_charts(df, types=types) == []


def test_suggest_infers_types_when_not_provided() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "v": [1, 2],
        }
    )
    specs = suggest_charts(df)
    assert any(s["type"] == "line" for s in specs)


def test_suggest_runs_on_dirty_fixture(dirty_df: pd.DataFrame) -> None:
    specs = suggest_charts(dirty_df, max_charts=8)
    # Dirty fixture has 1 datetime + 1 categorical + 1 boolean + numerics
    assert any(s["type"] == "line" for s in specs)
    assert any(s["type"] == "pie" for s in specs)
    assert len(specs) <= 8


def test_each_spec_has_required_fields() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "v": [1, 2],
            "c": ["a", "b"],
        }
    )
    types = {"d": "datetime", "v": "integer", "c": "categorical"}
    specs = suggest_charts(df, types=types)
    for s in specs:
        assert {"id", "type", "title", "x_column", "aggregation", "plotly_spec"} <= s.keys()
        assert s["type"] in {"line", "bar", "pie"}
        assert s["aggregation"] in {"sum", "mean", "count", "median", "min", "max"}
        assert "data" in s["plotly_spec"]
        assert "layout" in s["plotly_spec"]


def test_suggest_sorted_by_interestingness() -> None:
    # Two lines: one with high variance, one with constant values. The high-variance
    # line should rank first.
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "high_var": [1, 100, 2, 99],
            "flat": [5, 5, 5, 5],
        }
    )
    types = {"d": "datetime", "high_var": "integer", "flat": "integer"}
    specs = suggest_charts(df, types=types, max_charts=10)
    line_specs = [s for s in specs if s["type"] == "line"]
    assert line_specs[0]["y_column"] == "high_var"
