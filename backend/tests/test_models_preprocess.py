"""Tests for app.models.preprocess — ColumnTransformer builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.preprocess import build_preprocessor, split_feature_columns


def test_split_feature_columns_drops_text_and_datetime() -> None:
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": ["x", "y", "x"],  # categorical
            "c": [1.0, 2.0, 3.0],
            "d": ["hello", "world", "foo"],  # text
            "e": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "f": [True, False, True],
        }
    )
    types = {
        "a": "integer",
        "b": "categorical",
        "c": "float",
        "d": "text",
        "e": "datetime",
        "f": "boolean",
    }
    num, cat = split_feature_columns(df, types=types)
    assert num == ["a", "c"]
    assert cat == ["b", "f"]


def test_split_feature_columns_infers_when_types_missing() -> None:
    # Use enough rows so the categorical ratio threshold (0.5) is met.
    df = pd.DataFrame(
        {
            "a": list(range(20)),
            "b": (["x", "y"] * 10),
            "c": ["a long string of text here"] * 20,
        }
    )
    num, cat = split_feature_columns(df)
    assert num == ["a"]
    assert cat == ["b"]


def test_build_preprocessor_handles_numeric_only() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, 5.0]})
    pre = build_preprocessor(["a"], [])
    out = pre.fit_transform(df)
    # Imputed 4 missing-free values + median-imputed; output is 2D array.
    assert out.shape == (5, 1)
    # No NaNs in output
    assert not np.isnan(out).any()


def test_build_preprocessor_handles_categorical_only() -> None:
    df = pd.DataFrame({"g": ["a", "b", "a", "b", "c"]})
    pre = build_preprocessor([], ["g"])
    out = pre.fit_transform(df)
    # 3 categories → 3 one-hot columns
    assert out.shape == (5, 3)


def test_build_preprocessor_handles_mixed() -> None:
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "g": ["a", "b", "a", "b", "a"],
        }
    )
    pre = build_preprocessor(["a"], ["g"])
    out = pre.fit_transform(df)
    # 1 numeric (scaled) + 2 one-hot columns for 'a'/'b'
    assert out.shape == (5, 3)


def test_build_preprocessor_handles_unknown_category_at_predict() -> None:
    df_train = pd.DataFrame({"g": ["a", "b", "a", "b"]})
    pre = build_preprocessor([], ["g"]).fit(df_train)
    df_test = pd.DataFrame({"g": ["a", "c"]})  # 'c' was unseen
    out = pre.transform(df_test)
    # One-hot with handle_unknown='ignore' — 'c' should map to all zeros.
    assert out.shape == (2, 2)
    # 'c' should be the all-zero row
    assert np.array_equal(out[1], np.zeros(2))


def test_build_preprocessor_handles_nulls_in_numeric() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, None, 4.0, None, 6.0]})
    pre = build_preprocessor(["a"], [])
    out = pre.fit_transform(df)
    # No NaNs in transformed output
    assert not np.isnan(out).any()
    # Output has 6 rows
    assert out.shape[0] == 6


def test_build_preprocessor_empty_lists_returns_drop_transformer() -> None:
    pre = build_preprocessor([], [])
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = pre.fit_transform(df)
    # Drop everything; result has shape (3, 0)
    assert out.shape == (3, 0)
