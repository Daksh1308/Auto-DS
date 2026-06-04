"""Tests for app.models.train — train + evaluate orchestrator."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import make_classification, make_regression

from app.models.problem import ProblemInfo
from app.models.train import build_comparison_plotly_spec, train_and_evaluate


def _regression_df() -> pd.DataFrame:
    X, y = make_regression(
        n_samples=120, n_features=3, noise=5.0, random_state=42
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3"])
    df["y"] = y
    df["cat"] = ["a", "b"] * 60
    return df


def _classification_df() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=150,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4", "f5"])
    df["y"] = y
    df["cat"] = ["x", "y"] * 75
    return df


def test_regression_train_returns_three_models() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3", "cat"],
    )
    assert report.error is None
    assert len(report.models) == 3
    model_names = [m.name for m in report.models]
    assert "LinearRegression" in model_names
    assert "RandomForestRegressor" in model_names
    assert "HistGradientBoostingRegressor" in model_names


def test_regression_metrics_have_r2_mae_rmse() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    for m in report.models:
        assert "r2" in m.metrics
        assert "mae" in m.metrics
        assert "rmse" in m.metrics


def test_exactly_one_winner_is_marked() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    winners = [m for m in report.models if m.is_winner]
    assert len(winners) == 1


def test_classification_train_returns_three_models() -> None:
    df = _classification_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(
            kind="binary_classification", reason="test", n_classes=2
        ),
        features=["f1", "f2", "f3", "f4", "f5", "cat"],
    )
    assert report.error is None
    assert len(report.models) == 3
    model_names = [m.name for m in report.models]
    assert "LogisticRegression" in model_names
    assert "RandomForestClassifier" in model_names
    assert "HistGradientBoostingClassifier" in model_names


def test_classification_metrics_have_accuracy_and_f1() -> None:
    df = _classification_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(
            kind="binary_classification", reason="test", n_classes=2
        ),
        features=["f1", "f2", "f3", "f4", "f5"],
    )
    for m in report.models:
        assert "accuracy" in m.metrics
        assert "f1_macro" in m.metrics
        # ROC AUC only for binary + when predict_proba works
        if "roc_auc" in m.metrics:
            assert 0.0 <= m.metrics["roc_auc"] <= 1.0


def test_classification_reasonable_accuracy_on_synthetic() -> None:
    """The synthetic dataset is well-separable; expect accuracy > 0.7 for at
    least one model."""
    df = _classification_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(
            kind="binary_classification", reason="test", n_classes=2
        ),
        features=["f1", "f2", "f3", "f4", "f5"],
    )
    best = max(m.metrics.get("accuracy", 0) for m in report.models)
    assert best > 0.7


def test_missing_target_column_returns_error() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="nonexistent",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2"],
    )
    assert report.error is not None
    assert "nonexistent" in report.error


def test_missing_feature_column_returns_error() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "nope"],
    )
    assert report.error is not None
    assert "nope" in report.error


def test_no_features_returns_error() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=[],
    )
    assert report.error is not None
    assert "No usable features" in report.error


def test_too_small_dataset_returns_error() -> None:
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0, 5.0], "f": [1, 1, 1, 1, 1]})
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=5),
        features=["f"],
    )
    assert report.error is not None
    assert "too small" in report.error.lower()


def test_null_target_rows_are_dropped() -> None:
    df = _regression_df()
    df.loc[0:10, "y"] = None  # Drop 11 rows; 109 left
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    assert report.error is None
    assert report.rows_used == 109
    assert report.rows_total == 109


def test_downsampling_flag_triggers() -> None:
    # Make a dataset bigger than MAX_ROWS (50_000) by duplicating.
    df = _regression_df()
    big = pd.concat([df] * 500, ignore_index=True)  # 60,000 rows
    assert len(big) > 50_000
    report = train_and_evaluate(
        big,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    assert report.downsampled is True
    assert report.rows_total == 60_000
    assert report.rows_used == 50_000
    assert report.note is not None
    assert "Downsampled" in report.note


def test_comparison_plotly_spec_for_regression() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    spec = build_comparison_plotly_spec(report)
    assert spec is not None
    assert spec["data"][0]["type"] == "bar"
    assert len(spec["data"][0]["x"]) == 3
    # Y values should be in [0, 1] (R²)
    for y in spec["data"][0]["y"]:
        assert -1.0 <= y <= 1.0


def test_comparison_plotly_spec_for_classification() -> None:
    df = _classification_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(
            kind="binary_classification", reason="test", n_classes=2
        ),
        features=["f1", "f2", "f3", "f4", "f5"],
    )
    spec = build_comparison_plotly_spec(report)
    assert spec is not None
    assert spec["data"][0]["type"] == "bar"


def test_winner_metric_returns_name_and_metric() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    name, metric = report.winner_metric()
    assert name
    assert metric == "r2"


def test_train_seconds_recorded() -> None:
    df = _regression_df()
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(kind="regression", reason="test", n_classes=120),
        features=["f1", "f2", "f3"],
    )
    for m in report.models:
        assert m.train_seconds >= 0.0


def test_multiclass_classification_works() -> None:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=180,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4", "f5", "f6"])
    df["y"] = y
    report = train_and_evaluate(
        df,
        target="y",
        problem=ProblemInfo(
            kind="multiclass_classification", reason="test", n_classes=3
        ),
        features=["f1", "f2", "f3", "f4", "f5", "f6"],
    )
    assert report.error is None
    assert len(report.models) == 3
    for m in report.models:
        assert "accuracy" in m.metrics
        assert "f1_macro" in m.metrics
