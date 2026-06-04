"""Tests for app.cleaner.type_detect."""

from __future__ import annotations

import pandas as pd

from app.cleaner.type_detect import infer_column_type, infer_column_types


def test_infer_integer() -> None:
    assert infer_column_type(pd.Series([1, 2, 3, 4])) == "integer"


def test_infer_float() -> None:
    assert infer_column_type(pd.Series([1.5, 2.0, 3.7])) == "float"


def test_infer_datetime_from_string() -> None:
    s = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"])
    assert infer_column_type(s) == "datetime"


def test_infer_datetime_with_some_unparseable() -> None:
    s = pd.Series(["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "nope"])
    assert infer_column_type(s) == "datetime"


def test_infer_boolean() -> None:
    s = pd.Series(["yes", "no", "yes", "no"])
    assert infer_column_type(s) == "boolean"


def test_infer_categorical() -> None:
    s = pd.Series(["NYC"] * 50 + ["LA"] * 30 + ["Chicago"] * 20)
    assert infer_column_type(s) == "categorical"


def test_infer_text_for_high_cardinality_object() -> None:
    s = pd.Series([f"user_{i}@example.com" for i in range(50)])
    assert infer_column_type(s) == "text"


def test_infer_column_types_returns_mapping() -> None:
    rows = 50
    df = pd.DataFrame(
        {
            "n": list(range(1, rows + 1)),
            "x": [i + 0.5 for i in range(rows)],
            "d": [f"2024-01-{(i % 28) + 1:02d}" for i in range(rows)],
            "c": ["a" if i % 2 == 0 else "b" for i in range(rows)],
            "t": [f"u{i}@example.com" for i in range(rows)],
        }
    )
    types = infer_column_types(df)
    assert types["n"] == "integer"
    assert types["x"] == "float"
    assert types["d"] == "datetime"
    assert types["c"] == "categorical"
    assert types["t"] == "text"
