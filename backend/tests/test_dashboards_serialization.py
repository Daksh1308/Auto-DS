"""Tests for app.dashboards.serialization."""

from __future__ import annotations

import math
from datetime import datetime

import numpy as np
import pandas as pd

from app.dashboards.serialization import (
    dataframe_to_columns,
    dataframe_to_records,
    sample_dataframe,
)


def test_dataframe_to_columns_returns_list_of_strings() -> None:
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert dataframe_to_columns(df) == ["a", "b"]


def test_dataframe_to_records_handles_nan() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    records = dataframe_to_records(df)
    assert records[0]["x"] == 1.0
    assert records[1]["x"] is None
    assert records[2]["x"] == 3.0


def test_dataframe_to_records_handles_infinity() -> None:
    df = pd.DataFrame({"x": [1.0, math.inf, -math.inf]})
    records = dataframe_to_records(df)
    assert records[1]["x"] is None
    assert records[2]["x"] is None


def test_dataframe_to_records_handles_timestamp() -> None:
    df = pd.DataFrame({"d": pd.to_datetime(["2024-01-01", "2024-02-01"])})
    records = dataframe_to_records(df)
    assert records[0]["d"] == "2024-01-01T00:00:00"
    assert records[1]["d"] == "2024-02-01T00:00:00"


def test_dataframe_to_records_handles_numpy_types() -> None:
    df = pd.DataFrame(
        {
            "i": np.array([1, 2, 3], dtype=np.int64),
            "f": np.array([1.5, 2.5, 3.5], dtype=np.float64),
            "b": np.array([True, False, True]),
        }
    )
    records = dataframe_to_records(df)
    assert records[0] == {"i": 1, "f": 1.5, "b": True}


def test_sample_dataframe_returns_full_when_under_limit() -> None:
    df = pd.DataFrame({"x": range(10)})
    out = sample_dataframe(df, limit=100)
    assert len(out) == 10


def test_sample_dataframe_is_deterministic() -> None:
    df = pd.DataFrame({"x": range(100)})
    a = sample_dataframe(df, limit=10)
    b = sample_dataframe(df, limit=10)
    assert a["x"].tolist() == b["x"].tolist()


def test_sample_dataframe_caps_at_limit() -> None:
    df = pd.DataFrame({"x": range(500)})
    out = sample_dataframe(df, limit=50)
    assert len(out) == 50


def test_sample_dataframe_preserves_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    out = sample_dataframe(df, limit=2)
    assert list(out.columns) == ["a", "b"]
