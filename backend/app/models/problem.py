"""Detect the ML problem type from a target Series.

A target column is classified as one of:

* ``"regression"`` — continuous numeric with enough cardinality to not be a
  category. R²/MAE/RMSE are the headline metrics.
* ``"binary_classification"`` — two unique non-null values, regardless of
  dtype. Accuracy/F1/ROC AUC are the headline metrics.
* ``"multiclass_classification"`` — between 3 and ``MAX_CLASSIFICATION_CARDINALITY``
  unique non-null values and dtype is integer / boolean / categorical / small float.
  Accuracy/F1 (macro) are the headline metrics.
* ``"unsupported"`` — text / datetime / a single unique value / float with
  very low cardinality (likely an integer column read as float).

The heuristic is deliberately conservative: when in doubt, prefer
``"regression"`` over ``"unsupported"`` for float columns and prefer
``"unsupported"`` over ``"multiclass_classification"`` for ambiguous cases
(very high cardinality, text, datetime).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MAX_CLASSIFICATION_CARDINALITY = 20
MIN_CLASSES_FOR_CLASSIFICATION = 2


@dataclass(frozen=True)
class ProblemInfo:
    kind: str  # "regression" | "binary_classification" | "multiclass_classification" | "unsupported"
    reason: str
    n_classes: int | None = None


def detect_problem_type(
    series: pd.Series, detected_type: str | None = None
) -> ProblemInfo:
    """Return a :class:`ProblemInfo` for ``series``.

    ``detected_type`` is the value returned by
    :func:`app.cleaner.type_detect.infer_column_type` — using it lets the
    caller skip the dtype detection and pass the canonical label.
    """
    non_null = series.dropna()
    if non_null.empty:
        return ProblemInfo(
            kind="unsupported",
            reason="Target column has no non-null values",
            n_classes=0,
        )

    unique_values = non_null.unique()
    n_unique = len(unique_values)

    if detected_type is None:
        # Cheap fallback for callers that don't have the type_detect label.
        from ..cleaner.type_detect import infer_column_type

        detected_type = infer_column_type(series)

    if detected_type == "datetime":
        return ProblemInfo(
            kind="unsupported",
            reason="Datetime targets are not supported",
            n_classes=n_unique,
        )

    if detected_type == "text":
        return ProblemInfo(
            kind="unsupported",
            reason="Text targets are not supported",
            n_classes=n_unique,
        )

    if n_unique < MIN_CLASSES_FOR_CLASSIFICATION:
        return ProblemInfo(
            kind="unsupported",
            reason=f"Target has only {n_unique} unique value(s); need at least 2",
            n_classes=n_unique,
        )

    if n_unique == 2:
        return ProblemInfo(
            kind="binary_classification",
            reason="Target has exactly 2 unique values",
            n_classes=2,
        )

    # Boolean / categorical / small integer (≤ MAX_CLASSIFICATION_CARDINALITY) → classification.
    if detected_type in {"boolean", "categorical"}:
        if n_unique <= MAX_CLASSIFICATION_CARDINALITY:
            return ProblemInfo(
                kind="multiclass_classification",
                reason=(
                    f"Target is {detected_type} with {n_unique} classes "
                    f"(≤ {MAX_CLASSIFICATION_CARDINALITY})"
                ),
                n_classes=n_unique,
            )
        return ProblemInfo(
            kind="unsupported",
            reason=(
                f"Target is {detected_type} with {n_unique} classes "
                f"(> {MAX_CLASSIFICATION_CARDINALITY})"
            ),
            n_classes=n_unique,
        )

    if detected_type == "integer":
        if n_unique <= MAX_CLASSIFICATION_CARDINALITY:
            return ProblemInfo(
                kind="multiclass_classification",
                reason=(
                    f"Integer target with {n_unique} unique values "
                    f"(≤ {MAX_CLASSIFICATION_CARDINALITY})"
                ),
                n_classes=n_unique,
            )
        return ProblemInfo(
            kind="regression",
            reason=(
                f"Integer target with {n_unique} unique values "
                f"(> {MAX_CLASSIFICATION_CARDINALITY})"
            ),
            n_classes=n_unique,
        )

    if detected_type == "float":
        if n_unique <= MAX_CLASSIFICATION_CARDINALITY:
            # A float column with very few unique values is usually a cast
            # integer. Be conservative: treat as classification only when
            # values look like clean class labels.
            try:
                pd.testing.assert_index_equal(
                    pd.Index(sorted(unique_values)),
                    pd.Index([float(i) for i in range(n_unique)]),
                )
                looks_like_classes = True
            except AssertionError:
                looks_like_classes = False
            if looks_like_classes:
                return ProblemInfo(
                    kind="multiclass_classification",
                    reason=(
                        f"Float target with {n_unique} unique values that look "
                        "like class labels"
                    ),
                    n_classes=n_unique,
                )
        return ProblemInfo(
            kind="regression",
            reason="Continuous numeric target",
            n_classes=n_unique,
        )

    # Anything else (defensive — type_detect shouldn't produce this)
    return ProblemInfo(
        kind="unsupported",
        reason=f"Unrecognized target type '{detected_type}'",
        n_classes=n_unique,
    )


def is_supported(info: ProblemInfo) -> bool:
    return info.kind in {"regression", "binary_classification", "multiclass_classification"}
