"""IQR-based outlier detection for numeric columns."""

from __future__ import annotations

import pandas as pd


def iqr_outlier_count(series: pd.Series) -> int:
    """Return the number of values outside ``[Q1 - 1.5*IQR, Q3 + 1.5*IQR]``."""
    non_null = series.dropna()
    if len(non_null) < 4:
        return 0
    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return int(((non_null < lower) | (non_null > upper)).sum())
