"""Tests for app.insights.sentences."""

from __future__ import annotations

from app.insights.sentences import MAX_SENTENCES, render_sentences


def test_trend_sentence_uses_increase_verb() -> None:
    trends = [
        {
            "column": "sales",
            "period_col": "date",
            "first_period": "2024-Q1",
            "last_period": "2024-Q4",
            "first_value": 100.0,
            "last_value": 123.0,
            "pct_change": 23.0,
            "direction": "up",
        }
    ]
    sentences = render_sentences(trends, [], [], {})
    assert sentences[0] == "sales increased 23.0% from 2024-Q1 to 2024-Q4."


def test_trend_sentence_uses_decrease_verb_for_negative() -> None:
    trends = [
        {
            "column": "revenue",
            "period_col": "date",
            "first_period": "2024-Q1",
            "last_period": "2024-Q4",
            "first_value": 200.0,
            "last_value": 100.0,
            "pct_change": -50.0,
            "direction": "down",
        }
    ]
    sentences = render_sentences(trends, [], [], {})
    assert "decreased 50.0%" in sentences[0]


def test_correlation_sentence_uses_strength_word() -> None:
    correlations = [{"a": "x", "b": "y", "r": 0.82}]
    sentences = render_sentences([], correlations, [], {})
    assert "strongly" in sentences[0]
    assert "positively" in sentences[0]
    assert "0.82" in sentences[0]


def test_correlation_negative_uses_negatively() -> None:
    correlations = [{"a": "x", "b": "y", "r": -0.9}]
    sentences = render_sentences([], correlations, [], {})
    assert "negatively" in sentences[0]


def test_category_sentence_uses_higher() -> None:
    breakdowns = [
        {
            "category_col": "region",
            "numeric_col": "sales",
            "top": {"value": "N", "mean": 100.0},
            "bottom": {"value": "S", "mean": 50.0},
            "pct_diff": 100.0,
        }
    ]
    sentences = render_sentences([], [], breakdowns, {})
    assert "'N' has 100.0% higher average sales than 'S'" in sentences[0]


def test_distribution_sentence_flags_high_skew() -> None:
    columns = {
        "x": {
            "type": "float",
            "stats": {"count": 100, "skew": 2.5, "outliers_iqr": 1},
        }
    }
    sentences = render_sentences([], [], [], columns)
    assert any("right-skewed" in s for s in sentences)


def test_distribution_sentence_flags_outliers() -> None:
    columns = {
        "x": {
            "type": "integer",
            "stats": {"count": 100, "skew": 0.0, "outliers_iqr": 10},
        }
    }
    sentences = render_sentences([], [], [], columns)
    assert any("IQR-outlier" in s for s in sentences)


def test_distribution_sentence_skips_clean_column() -> None:
    columns = {
        "x": {
            "type": "float",
            "stats": {"count": 100, "skew": 0.2, "outliers_iqr": 1},
        }
    }
    sentences = render_sentences([], [], [], columns)
    assert all("skew" not in s and "outlier" not in s.lower() for s in sentences)


def test_caps_at_max_sentences() -> None:
    trends = [
        {
            "column": f"c{i}",
            "period_col": "d",
            "first_period": "2024-Q1",
            "last_period": "2024-Q4",
            "first_value": 1.0,
            "last_value": 2.0,
            "pct_change": 100.0,
            "direction": "up",
        }
        for i in range(20)
    ]
    sentences = render_sentences(trends, [], [], {})
    assert len(sentences) <= MAX_SENTENCES
    assert len(sentences) == MAX_SENTENCES


def test_ranking_order_trends_first() -> None:
    trends = [
        {
            "column": "sales",
            "period_col": "d",
            "first_period": "2024-Q1",
            "last_period": "2024-Q4",
            "first_value": 1.0,
            "last_value": 2.0,
            "pct_change": 100.0,
            "direction": "up",
        }
    ]
    correlations = [{"a": "x", "b": "y", "r": 0.9}]
    sentences = render_sentences(trends, correlations, [], {})
    assert sentences[0].startswith("sales")
    assert "correlated" in sentences[1]
