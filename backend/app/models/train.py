"""Train + evaluate a small set of sklearn models against the cleaned DataFrame.

This is the "AutoML lite" orchestrator. The pipeline per model is:

    ColumnTransformer(preprocess) -> Estimator

For each model we record:

* per-metric scores (R² / MAE / RMSE for regression, accuracy / F1 macro
  / ROC AUC for classification)
* training wall time in seconds

The best model per problem is marked with ``is_winner: True``. The choice of
"winner" metric is the one most users will care about for that problem:
R² for regression, accuracy for classification (with ROC AUC used to break
ties for binary classification).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from ..cleaner.type_detect import infer_column_types
from .preprocess import build_preprocessor, split_feature_columns
from .problem import ProblemInfo
from .registry import MODELS_BY_PROBLEM


RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_ROWS = 50_000
MIN_ROWS = 20


@dataclass
class ModelResult:
    name: str
    metrics: dict[str, float]
    train_seconds: float
    is_winner: bool = False
    error: str | None = None


@dataclass
class MLReport:
    target: str
    problem: ProblemInfo
    features: list[str]
    rows_used: int
    rows_total: int
    downsampled: bool
    models: list[ModelResult] = field(default_factory=list)
    note: str | None = None
    error: str | None = None

    def winner_metric(self) -> tuple[str, str]:
        """Return (model_name, metric_name) for the best model."""
        if not self.models:
            return ("", "")
        if self.problem.kind == "regression":
            metric = "r2"
            best = max(self.models, key=lambda m: m.metrics.get(metric, -math.inf))
            return (best.name, metric)
        # Classification: prefer accuracy, fall back to f1_macro.
        for metric in ("accuracy", "f1_macro"):
            best = max(
                (m for m in self.models if metric in m.metrics),
                key=lambda m: m.metrics.get(metric, -math.inf),
                default=None,
            )
            if best is not None:
                return (best.name, metric)
        return (self.models[0].name, "")


def _select_rows(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, bool, int]:
    """Drop null-target rows, then downsample to ``MAX_ROWS`` if needed."""
    cleaned = df.dropna(subset=[target])
    if len(cleaned) > MAX_ROWS:
        sampled = cleaned.sample(n=MAX_ROWS, random_state=RANDOM_STATE)
        return sampled, True, len(cleaned)
    return cleaned, False, len(cleaned)


def _compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(rmse),
    }


def _compute_classification_metrics(
    y_true, y_pred, y_score, problem_kind: str
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if problem_kind == "binary_classification" and y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            # y_score might be a 2D array; take the positive class column.
            if hasattr(y_score, "shape") and len(y_score.shape) == 2 and y_score.shape[1] == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_score[:, 1]))
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    problem: ProblemInfo,
    features: Sequence[str],
) -> MLReport:
    """Train all registered models and return an :class:`MLReport`."""
    if target not in df.columns:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=0,
            rows_total=len(df),
            downsampled=False,
            error=f"Target column '{target}' not in DataFrame",
        )
    missing = [f for f in features if f not in df.columns]
    if missing:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=0,
            rows_total=len(df),
            downsampled=False,
            error=f"Unknown feature column(s): {', '.join(missing)}",
        )
    if not features:
        return MLReport(
            target=target,
            problem=problem,
            features=[],
            rows_used=0,
            rows_total=len(df),
            downsampled=False,
            error="No usable features (target excluded, no numeric or categorical columns left).",
        )
    if target in features:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=0,
            rows_total=len(df),
            downsampled=False,
            error=f"Target column '{target}' cannot also be a feature.",
        )

    used_df, downsampled, rows_total = _select_rows(df, target)
    if len(used_df) < MIN_ROWS:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=len(used_df),
            rows_total=rows_total,
            downsampled=downsampled,
            error=f"Dataset too small to train (need ≥ {MIN_ROWS} non-null target rows, got {len(used_df)}).",
        )

    feature_types = infer_column_types(used_df[list(features)])
    numeric_cols, categorical_cols = split_feature_columns(
        used_df[list(features)], types=feature_types
    )
    if not numeric_cols and not categorical_cols:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=len(used_df),
            rows_total=rows_total,
            downsampled=downsampled,
            error="No usable features (target excluded, no numeric or categorical columns left).",
        )

    X = used_df[list(features)]
    y = used_df[target]

    stratify = y if problem.kind.endswith("classification") else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=stratify,
        )
    except ValueError as e:
        return MLReport(
            target=target,
            problem=problem,
            features=list(features),
            rows_used=len(used_df),
            rows_total=rows_total,
            downsampled=downsampled,
            error=f"Failed to split data: {e}",
        )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model_factories = MODELS_BY_PROBLEM.get(problem.kind, [])
    results: list[ModelResult] = []

    for name, estimator in model_factories:
        pipe = Pipeline(steps=[("pre", preprocessor), ("model", estimator)])
        start = time.perf_counter()
        try:
            pipe.fit(X_train, y_train)
        except Exception as e:
            results.append(
                ModelResult(
                    name=name,
                    metrics={},
                    train_seconds=time.perf_counter() - start,
                    error=f"{type(e).__name__}: {e}",
                )
            )
            continue
        train_seconds = time.perf_counter() - start

        y_pred = pipe.predict(X_test)
        metrics: dict[str, float]
        if problem.kind == "regression":
            metrics = _compute_regression_metrics(y_test, y_pred)
        else:
            y_score: np.ndarray | None = None
            if problem.kind == "binary_classification" and hasattr(pipe, "predict_proba"):
                try:
                    y_score = pipe.predict_proba(X_test)
                except Exception:
                    y_score = None
            metrics = _compute_classification_metrics(
                y_test, y_pred, y_score, problem.kind
            )
        results.append(
            ModelResult(
                name=name,
                metrics=metrics,
                train_seconds=train_seconds,
            )
        )

    # Mark winners
    report = MLReport(
        target=target,
        problem=problem,
        features=list(features),
        rows_used=len(used_df),
        rows_total=rows_total,
        downsampled=downsampled,
        models=results,
        note=(
            f"Downsampled from {rows_total:,} to {len(used_df):,} rows for training."
            if downsampled
            else None
        ),
    )
    if results:
        winner_name, _ = report.winner_metric()
        for r in results:
            if r.name == winner_name and not r.error:
                r.is_winner = True
    return report


def build_comparison_plotly_spec(report: MLReport) -> dict | None:
    """Build a Plotly bar chart comparing models on the headline metric."""
    if not report.models:
        return None
    metric_name = report.winner_metric()[1]
    if not metric_name:
        return None
    # Filter to models that produced a value for the headline metric
    rows = [m for m in report.models if m.error is None and metric_name in m.metrics]
    if not rows:
        return None
    rows.sort(key=lambda m: m.metrics[metric_name], reverse=True)

    metric_labels = {
        "r2": "R²",
        "accuracy": "Accuracy",
        "f1_macro": "F1 (macro)",
    }
    title = f"{metric_labels.get(metric_name, metric_name)} by model"

    return {
        "data": [
            {
                "type": "bar",
                "x": [m.name for m in rows],
                "y": [m.metrics[metric_name] for m in rows],
                "text": [f"{m.metrics[metric_name]:.4f}" for m in rows],
                "textposition": "outside",
            }
        ],
        "layout": {
            "title": {"text": title},
            "yaxis": {"title": {"text": metric_labels.get(metric_name, metric_name)}},
            "xaxis": {"title": {"text": "Model"}},
            "margin": {"l": 50, "r": 20, "t": 50, "b": 80},
        },
    }
