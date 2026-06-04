"""End-to-end cleaning pipeline."""

from __future__ import annotations

import pandas as pd

from .missing import fill_missing
from .normalize import normalize_columns, strip_whitespace
from .report import CleaningReport, ColumnReport
from .type_detect import infer_column_types


def run_cleaning(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline on ``df`` and return (cleaned_df, report)."""
    report = CleaningReport(
        rows_in=len(df),
        cols_in=df.shape[1],
    )

    df = strip_whitespace(df)
    df, renamed = normalize_columns(df)
    report.renamed_columns = renamed

    before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    report.duplicates_removed = before_dedup - len(df)

    types = infer_column_types(df)
    nulls_before = {col: int(df[col].isna().sum()) for col in df.columns}

    df, actions, dropped = fill_missing(df, types)
    report.dropped_columns = dropped
    report.cols_out = df.shape[1]
    report.rows_out = len(df)

    report.columns = [
        ColumnReport(
            name=col,
            detected_type=types.get(col, "unknown"),
            nulls_before=nulls_before.get(col, 0),
            nulls_after=int(df[col].isna().sum()) if col in df.columns else 0,
            action=actions.get(col, "no-op"),
        )
        for col in (list(nulls_before.keys()))
    ]

    return df, report
