"""Tests for app.cleaner.missing."""

from __future__ import annotations

import pandas as pd

from app.cleaner.missing import HIGH_NULL_DROP_THRESHOLD, fill_missing


def test_fill_numeric_with_median() -> None:
    df = pd.DataFrame({"n": [1.0, 2.0, None, 4.0, 100.0]})
    types = {"n": "float"}
    out, actions, dropped = fill_missing(df, types)
    assert out["n"].isna().sum() == 0
    assert out["n"].iloc[2] == 3.0
    assert "median" in actions["n"]
    assert dropped == []


def test_fill_categorical_with_mode() -> None:
    df = pd.DataFrame({"city": ["NYC", "LA", "NYC", None, "NYC"]})
    types = {"city": "categorical"}
    out, actions, _ = fill_missing(df, types)
    assert out["city"].isna().sum() == 0
    assert out["city"].iloc[3] == "NYC"
    assert "mode" in actions["city"]


def test_fill_text_with_unknown_constant() -> None:
    df = pd.DataFrame({"bio": ["hi", None, "yo"]})
    types = {"bio": "text"}
    out, actions, _ = fill_missing(df, types)
    assert out["bio"].iloc[1] == "unknown"
    assert "unknown" in actions["bio"]


def test_fill_datetime_with_ffill() -> None:
    df = pd.DataFrame(
        {"d": pd.to_datetime(["2024-01-01", "2024-02-01", None, "2024-04-01"])}
    )
    types = {"d": "datetime"}
    out, actions, _ = fill_missing(df, types)
    assert out["d"].isna().sum() == 0
    assert out["d"].iloc[2] == pd.Timestamp("2024-02-01")
    assert "forward-filled" in actions["d"]


def test_drops_columns_above_threshold() -> None:
    n_rows = 10
    nulls = int(HIGH_NULL_DROP_THRESHOLD * n_rows) + 1
    df = pd.DataFrame(
        {
            "good": list(range(1, n_rows + 1)),
            "bad": [None] * n_rows,
        }
    )
    # Force `bad` to be 70% null by overwriting a few entries (still 10 rows total)
    df.loc[: nulls - 1, "bad"] = None
    df.loc[nulls:, "bad"] = 0.0
    types = {"good": "integer", "bad": "float"}
    out, actions, dropped = fill_missing(df, types)
    assert "bad" in dropped
    assert "bad" not in out.columns
    assert "good" in out.columns
    assert "dropped" in actions["bad"]
