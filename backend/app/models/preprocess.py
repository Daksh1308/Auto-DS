"""Build a :class:`ColumnTransformer` for ML feature preprocessing.

* numeric columns → median-impute + standardize
* categorical columns → most-frequent-impute + one-hot (handle unknowns)

Datetime columns are dropped (caller should exclude them from the feature
list beforehand) because AutoML-lite does not encode timestamps.
"""

from __future__ import annotations

from typing import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..cleaner.type_detect import infer_column_types


_NUMERIC_TYPES = {"integer", "float"}
_CATEGORICAL_TYPES = {"categorical", "boolean"}


def split_feature_columns(
    df: pd.DataFrame, types: dict[str, str] | None = None
) -> tuple[list[str], list[str]]:
    """Return ``(numeric_cols, categorical_cols)`` for ``df``.

    Datetime and text columns are dropped — they are not usable as features
    in this lightweight pipeline. Callers should not pass those columns in
    the first place; this is a safety net.
    """
    if types is None:
        types = infer_column_types(df)
    numeric: list[str] = []
    categorical: list[str] = []
    for col in df.columns:
        t = types.get(col, "text")
        if t in _NUMERIC_TYPES:
            numeric.append(col)
        elif t in _CATEGORICAL_TYPES:
            categorical.append(col)
        # datetime, text, anything else → dropped
    return numeric, categorical


def build_preprocessor(
    numeric_cols: Sequence[str], categorical_cols: Sequence[str]
) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, list(numeric_cols)))
    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipeline, list(categorical_cols)))
    if not transformers:
        return ColumnTransformer(
            transformers=[],
            remainder="drop",
        )
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# Make pandas importable for type hints in build_preprocessor signatures.
import pandas as pd  # noqa: E402  (placed here to avoid an import cycle)
