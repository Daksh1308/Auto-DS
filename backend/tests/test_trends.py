"""Tests for app.insights.trends."""

from __future__ import annotations

import pandas as pd

from app.insights.trends import _bin_to_4_periods, _label_for_date, compute_trends


def test_label_quarter() -> None:
    assert _label_for_date(pd.Timestamp("2024-01-15"), "quarter") == "2024-Q1"
    assert _label_for_date(pd.Timestamp("2024-04-01"), "quarter") == "2024-Q2"
    assert _label_for_date(pd.Timestamp("2024-12-31"), "quarter") == "2024-Q4"


def test_label_month() -> None:
    assert _label_for_date(pd.Timestamp("2024-03-15"), "month") == "2024-03"


def test_label_week() -> None:
    label = _label_for_date(pd.Timestamp("2024-03-15"), "week")
    assert label.startswith("2024-W")


def test_bin_to_4_periods_produces_four_labels() -> None:
    s = pd.to_datetime(
        ["2024-01-01", "2024-03-01", "2024-05-01", "2024-07-01", "2024-09-01"]
    )
    binned, labels, _kind = _bin_to_4_periods(s)
    assert labels is not None and len(labels) == 4
    assert binned is not None
    assert set(binned.dropna().unique()) <= set(labels)


def test_trend_reports_pct_change() -> None:
    # Sales falls sharply from the first period to the last.
    # With 4 equal-width periods across 8 monthly dates, periods each get 2 rows.
    # We arrange values so the first period is high and the last is low.
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-15",
                    "2024-02-15",
                    "2024-04-15",
                    "2024-05-15",
                    "2024-07-15",
                    "2024-08-15",
                    "2024-10-15",
                    "2024-11-15",
                ]
            ),
            "sales": [100, 100, 50, 50, 25, 25, 5, 5],
        }
    )
    types = {"date": "datetime", "sales": "integer"}
    out = compute_trends(df, types)
    assert len(out) == 1
    trend = out[0]
    assert trend["column"] == "sales"
    assert trend["direction"] == "down"
    assert trend["pct_change"] < -50  # 200 -> 10 is a 95% drop


def test_trend_no_datetime_returns_empty() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert compute_trends(df, {"x": "integer"}) == []


def test_trend_picks_widest_datetime() -> None:
    df = pd.DataFrame(
        {
            "narrow": pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
            "wide": pd.to_datetime(
                ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
            ),
            "value": [1, 2, 3, 4],
        }
    )
    types = {"narrow": "datetime", "wide": "datetime", "value": "integer"}
    out = compute_trends(df, types)
    assert out and out[0]["period_col"] == "wide"


def test_trend_skips_all_zero_columns() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01"]
            ),
            "zero": [0, 0, 0, 0],
        }
    )
    types = {"date": "datetime", "zero": "integer"}
    assert compute_trends(df, types) == []
