"""Tests for the full cleaning pipeline."""

from __future__ import annotations

import pandas as pd

from app.cleaner.pipeline import run_cleaning


def test_pipeline_runs_on_minimal_frame() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3]})
    cleaned, report = run_cleaning(df)
    assert report.rows_in == 4
    assert report.rows_out == 3
    assert report.duplicates_removed == 1
    assert cleaned["a"].tolist() == [1, 2, 3]


def test_pipeline_normalizes_and_dedupes(dirty_df: pd.DataFrame) -> None:
    cleaned, report = run_cleaning(dirty_df)
    # First and last row are duplicates of "Alice Smith" -> 1 duplicate removed
    assert report.duplicates_removed == 1
    assert all(c.islower() and " " not in c for c in cleaned.columns)


def test_pipeline_fills_missing_values(dirty_df: pd.DataFrame) -> None:
    cleaned, _ = run_cleaning(dirty_df)
    assert cleaned.isna().sum().sum() == 0


def test_pipeline_drops_high_null_column(dirty_df: pd.DataFrame) -> None:
    _, report = run_cleaning(dirty_df)
    assert "mostly_empty" in report.dropped_columns


def test_pipeline_produces_type_report(dirty_df: pd.DataFrame) -> None:
    _, report = run_cleaning(dirty_df)
    types_by_name = {c.name: c.detected_type for c in report.columns}
    assert types_by_name["age"] == "integer"
    assert types_by_name["city"] == "categorical"
    assert types_by_name["sign_up_date"] == "datetime"
    assert types_by_name["subscribed"] == "boolean"


def test_pipeline_records_renames(dirty_df: pd.DataFrame) -> None:
    _, report = run_cleaning(dirty_df)
    assert report.renamed_columns.get("Customer ID") == "customer_id"
    assert report.renamed_columns.get("Full Name") == "full_name"
    assert report.renamed_columns.get("SignUp Date") == "sign_up_date"
