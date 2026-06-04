"""Fixed, lightweight model registry. No HPO, no ensembling."""

from __future__ import annotations

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression


RANDOM_STATE = 42


def _regression_models() -> list[tuple[str, BaseEstimator]]:
    return [
        ("LinearRegression", LinearRegression()),
        (
            "RandomForestRegressor",
            RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
            ),
        ),
        (
            "HistGradientBoostingRegressor",
            HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        ),
    ]


def _classification_models() -> list[tuple[str, BaseEstimator]]:
    return [
        ("LogisticRegression", LogisticRegression(max_iter=1000)),
        (
            "RandomForestClassifier",
            RandomForestClassifier(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=1
            ),
        ),
        (
            "HistGradientBoostingClassifier",
            HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        ),
    ]


MODELS_BY_PROBLEM: dict[str, list[tuple[str, BaseEstimator]]] = {
    "regression": _regression_models(),
    "binary_classification": _classification_models(),
    "multiclass_classification": _classification_models(),
}
