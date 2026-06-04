"""Tests for app.insights.category_breakdown."""

from __future__ import annotations

import pandas as pd

from app.insights.category_breakdown import compute_category_breakdowns


def test_top_vs_bottom_computed() -> None:
    df = pd.DataFrame(
        {
            "region": ["N", "N", "N", "S", "S", "S"],
            "sales": [100, 110, 90, 50, 60, 40],
        }
    )
    out = compute_category_breakdowns(df, {"region": "categorical", "sales": "integer"})
    assert len(out) == 1
    item = out[0]
    assert item["top"]["value"] == "N"
    assert item["bottom"]["value"] == "S"
    assert item["pct_diff"] > 0


def test_no_categoricals_returns_empty() -> None:
    df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    out = compute_category_breakdowns(df, {"x": "integer", "y": "integer"})
    assert out == []


def test_single_category_skipped() -> None:
    df = pd.DataFrame({"c": ["x", "x", "x"], "v": [1, 2, 3]})
    out = compute_category_breakdowns(df, {"c": "categorical", "v": "integer"})
    assert out == []


def test_boolean_treated_as_categorical() -> None:
    df = pd.DataFrame(
        {
            "sub": ["yes", "yes", "no", "no"],
            "spend": [100, 110, 50, 60],
        }
    )
    out = compute_category_breakdowns(df, {"sub": "boolean", "spend": "integer"})
    assert len(out) == 1
    assert out[0]["top"]["value"] in {"yes", "no"}


def test_sorted_by_abs_pct_diff() -> None:
    df = pd.DataFrame(
        {
            "a": ["x", "x", "x", "y", "y", "y", "z", "z", "z"] * 2,
            "m1": [1, 1, 1, 2, 2, 2, 3, 3, 3] * 2,
            "m2": [10, 10, 10, 1, 1, 1, 1, 1, 1] * 2,
        }
    )
    types = {"a": "categorical", "m1": "integer", "m2": "integer"}
    out = compute_category_breakdowns(df, types)
    diffs = [abs(item["pct_diff"]) for item in out]
    assert diffs == sorted(diffs, reverse=True)


def test_handles_nans_in_numeric() -> None:
    df = pd.DataFrame(
        {
            "c": ["a", "a", "b", "b", "a", "b"],
            "v": [1, None, 3, None, 2, 4],
        }
    )
    out = compute_category_breakdowns(df, {"c": "categorical", "v": "float"})
    assert len(out) == 1
    # a: [1, 2] mean=1.5; b: [3, 4] mean=3.5
    assert out[0]["top"]["value"] == "b"
    assert out[0]["bottom"]["value"] == "a"
