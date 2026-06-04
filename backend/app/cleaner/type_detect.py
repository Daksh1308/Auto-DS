"""Column-type inference for tabular data.

Detection order (highest precedence first):
    datetime -> boolean -> integer -> float -> categorical -> text

A column is labelled ``datetime`` if ``pd.to_datetime(..., errors='coerce')``
parses at least ``DATETIME_PARSE_RATIO`` of the non-null values. Boolean
detection accepts common truthy/falsy spellings. Numeric detection uses
``pd.api.types.is_numeric_dtype`` after coercing to numeric. Categorical
applies only to object columns whose unique ratio is below
``CATEGORICAL_RATIO_THRESHOLD``. Everything else is ``text``.
"""

from __future__ import annotations

import pandas as pd


DATETIME_PARSE_RATIO = 0.8
CATEGORICAL_RATIO_THRESHOLD = 0.5
CATEGORICAL_MIN_UNIQUE = 2

_BOOLEAN_TRUE = {"true", "false", "yes", "no", "y", "n", "t", "f", "1", "0"}
_BOOLEAN_TRUE_VALUES = {"true", "yes", "y", "t", "1"}
_BOOLEAN_FALSE_VALUES = {"false", "no", "n", "f", "0"}


def _is_boolean(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    lowered = non_null.astype(str).str.strip().str.lower()
    return set(lowered.unique()).issubset(_BOOLEAN_TRUE)


def _is_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    ratio = parsed.notna().sum() / len(non_null)
    return ratio >= DATETIME_PARSE_RATIO


def _is_integer(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True
    # Float dtype often hides "integer" data (especially after CSV read
    # where NaNs force float64). Treat as integer when every non-null
    # value is a whole number.
    if pd.api.types.is_float_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return False
        return (non_null % 1 == 0).all()
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    if coerced.isna().any():
        return False
    return (coerced % 1 == 0).all()


def _is_float(series: pd.Series) -> bool:
    if pd.api.types.is_float_dtype(series):
        return True
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if non_null.empty:
        return False
    coerced = pd.to_numeric(non_null, errors="coerce")
    return coerced.notna().all() and not (coerced % 1 == 0).all()


def _is_categorical(series: pd.Series) -> bool:
    if series.dtype != object:
        return False
    non_null = series.dropna()
    if non_null.nunique() < CATEGORICAL_MIN_UNIQUE:
        return False
    return (non_null.nunique() / len(non_null)) < CATEGORICAL_RATIO_THRESHOLD


def infer_column_type(series: pd.Series) -> str:
    """Infer a single column's semantic type from a pandas Series."""
    if _is_boolean(series):
        return "boolean"
    if _is_datetime(series):
        return "datetime"
    if _is_integer(series):
        return "integer"
    if _is_float(series):
        return "float"
    if _is_categorical(series):
        return "categorical"
    return "text"


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return a mapping of column name -> inferred type for every column."""
    return {col: infer_column_type(df[col]) for col in df.columns}
