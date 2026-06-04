"""Tests for app.models.problem — ML problem-type detection."""

from __future__ import annotations

import pandas as pd

from app.models.problem import detect_problem_type, is_supported


def test_detects_binary_classification() -> None:
    s = pd.Series([0, 1, 0, 1, 1, 0] * 5)
    info = detect_problem_type(s, "integer")
    assert info.kind == "binary_classification"
    assert info.n_classes == 2
    assert is_supported(info)


def test_detects_binary_classification_with_booleans() -> None:
    s = pd.Series([True, False, True, False] * 5)
    info = detect_problem_type(s, "boolean")
    assert info.kind == "binary_classification"


def test_detects_multiclass_integer() -> None:
    s = pd.Series([0, 1, 2, 0, 1, 2] * 5)
    info = detect_problem_type(s, "integer")
    assert info.kind == "multiclass_classification"
    assert info.n_classes == 3


def test_detects_multiclass_categorical() -> None:
    s = pd.Series(["a", "b", "c", "a", "b", "c"] * 5)
    info = detect_problem_type(s, "categorical")
    assert info.kind == "multiclass_classification"
    assert info.n_classes == 3


def test_detects_regression_for_continuous_float() -> None:
    s = pd.Series([0.1, 1.2, 2.5, 3.7, 4.0, 5.3, 6.9, 7.1, 8.8, 9.4] * 5)
    info = detect_problem_type(s, "float")
    assert info.kind == "regression"


def test_detects_regression_for_high_cardinality_integer() -> None:
    s = pd.Series(list(range(100)))
    info = detect_problem_type(s, "integer")
    assert info.kind == "regression"
    assert info.n_classes == 100


def test_detects_unsupported_text() -> None:
    s = pd.Series(["apple", "banana", "cherry"] * 5)
    info = detect_problem_type(s, "text")
    assert info.kind == "unsupported"
    assert not is_supported(info)


def test_detects_unsupported_datetime() -> None:
    s = pd.to_datetime(["2020-01-01", "2020-12-31"] * 5)
    info = detect_problem_type(s, "datetime")
    assert info.kind == "unsupported"


def test_detects_unsupported_single_value() -> None:
    s = pd.Series([42] * 10)
    info = detect_problem_type(s, "integer")
    assert info.kind == "unsupported"
    assert "only 1 unique" in info.reason


def test_detects_unsupported_high_cardinality_categorical() -> None:
    s = pd.Series([f"c{i}" for i in range(50)])
    info = detect_problem_type(s, "categorical")
    assert info.kind == "unsupported"
    assert "50 classes" in info.reason


def test_detects_unsupported_null_target() -> None:
    s = pd.Series([None, None, None])
    info = detect_problem_type(s, "integer")
    assert info.kind == "unsupported"
    assert "no non-null" in info.reason


def test_binary_when_just_two_unique_non_null() -> None:
    s = pd.Series([1, 1, 1, 2, 2, 2, None, None])
    info = detect_problem_type(s, "integer")
    assert info.kind == "binary_classification"
    assert info.n_classes == 2


def test_unsupported_categorical_too_many_classes() -> None:
    s = pd.Series([f"c{i}" for i in range(30)] * 2)
    info = detect_problem_type(s, "categorical")
    assert info.kind == "unsupported"


def test_detects_multiclass_when_float_looks_like_classes() -> None:
    s = pd.Series([0.0, 1.0, 2.0, 0.0, 1.0, 2.0] * 5)
    info = detect_problem_type(s, "float")
    assert info.kind == "multiclass_classification"


def test_detects_regression_for_arbitrary_floats() -> None:
    s = pd.Series([0.1, 1.5, 2.3, 3.7] * 5)
    info = detect_problem_type(s, "float")
    assert info.kind == "regression"


def test_infers_type_when_not_given() -> None:
    """Calling without a detected_type still works (falls back to infer_column_type)."""
    s = pd.Series([0, 1, 0, 1] * 5)
    info = detect_problem_type(s)
    assert info.kind == "binary_classification"
