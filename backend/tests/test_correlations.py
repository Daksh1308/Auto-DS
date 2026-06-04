"""Tests for app.insights.correlations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.insights.correlations import compute_correlations


def test_strong_positive_correlation_detected() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    df = pd.DataFrame({"a": x, "b": x * 2 + rng.normal(0, 0.01, 200)})
    out = compute_correlations(df, ["a", "b"], min_abs_r=0.5)
    assert any(abs(item["r"]) > 0.99 for item in out)


def test_strong_negative_correlation_detected() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    df = pd.DataFrame({"a": x, "b": -x + rng.normal(0, 0.01, 200)})
    out = compute_correlations(df, ["a", "b"], min_abs_r=0.5)
    assert any(item["r"] < -0.99 for item in out)


def test_weak_correlation_filtered() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(0, 1, 200),  # independent -> r ~ 0
        }
    )
    out = compute_correlations(df, ["a", "b"], min_abs_r=0.5)
    assert out == []


def test_threshold_filters_moderate_correlation() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    df = pd.DataFrame(
        {"a": x, "b": x + rng.normal(0, 1.5, 500)}  # weak-moderate
    )
    weak = compute_correlations(df, ["a", "b"], min_abs_r=0.3)
    strict = compute_correlations(df, ["a", "b"], min_abs_r=0.9)
    assert len(weak) >= len(strict)


def test_returns_deduped_pairs() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8], "c": [3, 5, 7, 9]})
    out = compute_correlations(df, ["a", "b", "c"], min_abs_r=0.0)
    keys = [(item["a"], item["b"]) for item in out]
    assert len(keys) == len(set(keys))


def test_sorted_by_abs_r_descending() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 300)
    df = pd.DataFrame({"a": x, "b": x, "c": -x, "d": rng.normal(0, 1, 300)})
    out = compute_correlations(df, ["a", "b", "c", "d"], min_abs_r=0.0)
    abs_rs = [abs(item["r"]) for item in out]
    assert abs_rs == sorted(abs_rs, reverse=True)


def test_handles_fewer_than_two_columns() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert compute_correlations(df, ["a"], min_abs_r=0.0) == []
