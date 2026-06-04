"""Tests for app.cleaner.normalize."""

from __future__ import annotations

import pandas as pd

from app.cleaner.normalize import normalize_columns, strip_whitespace


def test_strip_whitespace_trims_string_cells() -> None:
    df = pd.DataFrame({"name": ["  alice  ", "bob  ", "\tcarol\n"]})
    out = strip_whitespace(df)
    assert out["name"].tolist() == ["alice", "bob", "carol"]


def test_strip_whitespace_ignores_non_string_cells() -> None:
    df = pd.DataFrame({"n": [1, 2, 3], "x": [1.5, None, 2.5]})
    out = strip_whitespace(df)
    pd.testing.assert_frame_equal(out, df)


def test_normalize_columns_snake_case() -> None:
    df = pd.DataFrame(columns=["First Name", "Last Name", "Age (years)"])
    out, rename = normalize_columns(df)
    assert list(out.columns) == ["first_name", "last_name", "age_years"]
    assert rename == {
        "First Name": "first_name",
        "Last Name": "last_name",
        "Age (years)": "age_years",
    }


def test_normalize_columns_dedupes_collisions() -> None:
    df = pd.DataFrame(columns=["Name", "name", "NAME", "other"])
    out, rename = normalize_columns(df)
    assert list(out.columns) == ["name", "name_1", "name_2", "other"]
    assert rename == {"Name": "name", "name": "name_1", "NAME": "name_2", "other": "other"}


def test_normalize_columns_handles_camel_case() -> None:
    df = pd.DataFrame(columns=["customerID", "OrderTotal", "  weird-col "])
    out, _ = normalize_columns(df)
    assert list(out.columns) == ["customer_id", "order_total", "weird_col"]
