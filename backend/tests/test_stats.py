"""Tests for app.insights.stats."""

from __future__ import annotations

import pandas as pd

from app.insights.stats import compute_column_stats


def test_numeric_stats_shape() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    types = {"x": "integer"}
    out = compute_column_stats(df, types)
    assert out["x"]["type"] == "integer"
    s = out["x"]["stats"]
    assert s["count"] == 5
    assert s["mean"] == 3.0
    assert s["min"] == 1.0
    assert s["max"] == 5.0
    assert s["q25"] == 2.0
    assert s["q75"] == 4.0
    assert s["outliers_iqr"] == 0


def test_numeric_stats_detects_outliers() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 100]})  # 100 is well outside IQR
    out = compute_column_stats(df, {"x": "integer"})
    assert out["x"]["stats"]["outliers_iqr"] >= 1


def test_numeric_stats_handles_nans() -> None:
    df = pd.DataFrame({"x": [1, None, 3, None, 5]})
    out = compute_column_stats(df, {"x": "float"})
    assert out["x"]["stats"]["count"] == 3
    assert out["x"]["stats"]["mean"] == 3.0


def test_categorical_stats() -> None:
    df = pd.DataFrame({"c": ["a", "b", "a", "a", "c"]})
    out = compute_column_stats(df, {"c": "categorical"})
    s = out["c"]["stats"]
    assert s["count"] == 5
    assert s["unique"] == 3
    assert s["top"] == "a"
    assert s["top_count"] == 3
    assert s["top_pct"] == 60.0


def test_text_stats() -> None:
    df = pd.DataFrame({"t": ["hi", "hello", "yo"]})
    out = compute_column_stats(df, {"t": "text"})
    s = out["t"]["stats"]
    assert s["count"] == 3
    assert s["unique"] == 3
    assert s["max_length"] == 5.0


def test_datetime_stats() -> None:
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-31"])})
    out = compute_column_stats(df, {"d": "datetime"})
    s = out["d"]["stats"]
    assert s["count"] == 3
    assert s["span_days"] == 365


def test_empty_column_does_not_crash() -> None:
    df = pd.DataFrame({"x": [None, None, None]})
    out = compute_column_stats(df, {"x": "float"})
    assert out["x"]["stats"]["count"] == 0
