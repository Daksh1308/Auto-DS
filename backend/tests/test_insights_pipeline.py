"""End-to-end tests for the insights pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.insights.pipeline import run_insights


def test_pipeline_returns_report_shape() -> None:
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            "c": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    out = run_insights(df)
    d = out.to_dict()
    assert d["schema"]["rows"] == 10
    assert d["schema"]["cols"] == 3
    assert "x" in d["columns"]
    assert len(d["correlations"]) >= 1  # x and y are perfectly correlated


def test_pipeline_runs_on_dirty_fixture(dirty_df: pd.DataFrame) -> None:
    out = run_insights(dirty_df)
    d = out.to_dict()
    assert d["schema"]["rows"] == 12
    # No datetime column in the original dirty frame, so no trends here.
    # We test trends on a synthetic frame below.
    assert isinstance(d["insight_sentences"], list)


def test_pipeline_with_datetime_and_trend() -> None:
    rng = np.random.default_rng(0)
    n = 200
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    sales = np.linspace(100, 200, n) + rng.normal(0, 5, n)
    df = pd.DataFrame({"date": dates, "sales": sales})
    out = run_insights(df)
    trends = out.trends
    assert len(trends) == 1
    assert trends[0]["column"] == "sales"
    assert trends[0]["direction"] == "up"
    # 100 -> 200 over 4 equal-width periods means the last period is roughly
    # 175..200 (sum ~9375) and the first is 100..125 (sum ~5625) -> ~67% change
    assert trends[0]["pct_change"] > 50


def test_pipeline_sentences_respect_max() -> None:
    n = 50
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            **{f"v{i}": np.linspace(0, i + 1, n) for i in range(20)},
        }
    )
    out = run_insights(df)
    assert len(out.insight_sentences) <= 10


def test_pipeline_min_corr_filter() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 500),
            "b": rng.normal(0, 1, 500),  # independent
        }
    )
    out_strict = run_insights(df, min_abs_corr=0.9)
    assert out_strict.correlations == []
    out_loose = run_insights(df, min_abs_corr=0.0)
    # at default nothing may pass; we just check no exception
    assert isinstance(out_loose.correlations, list)
