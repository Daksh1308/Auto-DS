"""Tests for app.chat.schema — compact schema prompt builder."""

from __future__ import annotations

import pandas as pd

from app.chat.schema import build_schema_prompt


def test_schema_prompt_includes_row_count() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = build_schema_prompt(df)
    assert "3 rows" in out
    assert "1 columns" in out


def test_schema_prompt_describes_numeric_range() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4]})
    out = build_schema_prompt(df)
    assert "x (integer)" in out
    assert "range 1..4" in out


def test_schema_prompt_describes_categorical_values() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b", "b"]})
    out = build_schema_prompt(df)
    assert "g (" in out
    # values are surfaced
    assert "a" in out
    assert "b" in out


def test_schema_prompt_describes_datetime_range() -> None:
    df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2020-12-31"])})
    out = build_schema_prompt(df)
    assert "datetime" in out
    assert "2020-01-01" in out
    assert "2020-12-31" in out


def test_schema_prompt_includes_sample_rows() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = build_schema_prompt(df)
    assert '"a"' in out
    assert '"b"' in out
    assert "1" in out
    assert '"x"' in out


def test_schema_prompt_handles_empty_dataframe() -> None:
    df = pd.DataFrame({"a": []})
    out = build_schema_prompt(df)
    assert "0 rows" in out
    assert "1 columns" in out


def test_schema_prompt_truncates_text_sample() -> None:
    long_val = "x" * 100
    df = pd.DataFrame({"t": [long_val, "short"]})
    out = build_schema_prompt(df)
    assert "..." in out
