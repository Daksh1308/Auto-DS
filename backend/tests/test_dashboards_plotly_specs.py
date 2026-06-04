"""Tests for app.dashboards.plotly_specs."""

from __future__ import annotations

import pandas as pd

from app.dashboards.plotly_specs import aggregate, build_plotly_spec


def test_aggregate_count_by_category() -> None:
    df = pd.DataFrame({"c": ["a", "b", "a", "c", "b", "a"]})
    out = aggregate(df, "c", None, "count")
    assert set(out.columns) == {"c", "y"}
    counts = dict(zip(out["c"], out["y"]))
    assert counts == {"a": 3, "b": 2, "c": 1}


def test_aggregate_sum_by_category() -> None:
    df = pd.DataFrame({"c": ["a", "a", "b"], "v": [10, 20, 5]})
    out = aggregate(df, "c", "v", "sum")
    counts = dict(zip(out["c"], out["y"]))
    assert counts == {"a": 30, "b": 5}


def test_aggregate_mean_by_category() -> None:
    df = pd.DataFrame({"c": ["a", "a", "b"], "v": [10, 20, 5]})
    out = aggregate(df, "c", "v", "mean")
    counts = dict(zip(out["c"], out["y"]))
    assert counts == {"a": 15.0, "b": 5.0}


def test_aggregate_median_min_max() -> None:
    df = pd.DataFrame({"c": ["a", "a", "a", "b", "b"], "v": [1, 5, 9, 2, 8]})
    assert aggregate(df, "c", "v", "median")["y"].tolist() == [5.0, 5.0]
    assert aggregate(df, "c", "v", "min")["y"].tolist() == [1.0, 2.0]
    assert aggregate(df, "c", "v", "max")["y"].tolist() == [9.0, 8.0]


def test_aggregate_with_datetime_x() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "v": [1, 2, 4],
        }
    )
    out = aggregate(df, "d", "v", "sum")
    assert len(out) == 2
    assert out["y"].sum() == 7


def test_aggregate_missing_x_raises() -> None:
    df = pd.DataFrame({"c": ["a"]})
    try:
        aggregate(df, "missing", "c", "sum")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_aggregate_missing_y_raises() -> None:
    df = pd.DataFrame({"c": ["a"], "v": [1]})
    try:
        aggregate(df, "c", "missing", "sum")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_build_line_spec_shape() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
            "sales": [10, 20, 30],
        }
    )
    spec = build_plotly_spec("line", df, "d", "sales", "sum", "Sales over time")
    assert spec["data"][0]["type"] == "scatter"
    assert spec["data"][0]["mode"] == "lines+markers"
    assert spec["data"][0]["x"] == ["2024-01-01T00:00:00", "2024-02-01T00:00:00", "2024-03-01T00:00:00"]
    assert spec["data"][0]["y"] == [10, 20, 30]
    assert spec["layout"]["title"]["text"] == "Sales over time"


def test_build_bar_spec_shape() -> None:
    df = pd.DataFrame({"region": ["N", "S", "N", "E"], "sales": [10, 20, 30, 40]})
    spec = build_plotly_spec("bar", df, "region", "sales", "sum", "Sales by region")
    assert spec["data"][0]["type"] == "bar"
    # Sorted by y desc -> E:40, N:40, S:20
    assert spec["data"][0]["y"][0] == 40


def test_build_pie_spec_shape() -> None:
    df = pd.DataFrame({"c": ["a", "b", "a", "c", "b", "a"]})
    spec = build_plotly_spec("pie", df, "c", None, "count", "Distribution")
    assert spec["data"][0]["type"] == "pie"
    assert set(spec["data"][0]["labels"]) == {"a", "b", "c"}
    assert sum(spec["data"][0]["values"]) == 6


def test_build_empty_data_returns_placeholder() -> None:
    df = pd.DataFrame({"c": [], "v": []})
    spec = build_plotly_spec("line", df, "c", "v", "sum", "Empty")
    assert spec["data"] == []


def test_build_unknown_chart_type_raises() -> None:
    df = pd.DataFrame({"c": ["a"]})
    try:
        build_plotly_spec("scatter", df, "c", None, "count", "X")
    except ValueError:
        return
    raise AssertionError("expected ValueError")
