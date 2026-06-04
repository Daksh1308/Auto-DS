"""Insights pipeline orchestrator."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..cleaner.type_detect import infer_column_types
from .category_breakdown import compute_category_breakdowns
from .correlations import compute_correlations
from .sentences import render_sentences
from .stats import compute_column_stats
from .trends import compute_trends


_NUMERIC_TYPES = {"integer", "float"}


@dataclass
class InsightsReport:
    schema: dict = field(default_factory=dict)
    columns: dict = field(default_factory=dict)
    correlations: list = field(default_factory=list)
    trends: list = field(default_factory=list)
    category_breakdowns: list = field(default_factory=list)
    insight_sentences: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema": dict(self.schema),
            "columns": dict(self.columns),
            "correlations": list(self.correlations),
            "trends": list(self.trends),
            "category_breakdowns": list(self.category_breakdowns),
            "insight_sentences": list(self.insight_sentences),
        }


def _build_schema(df: pd.DataFrame, types: dict[str, str]) -> dict:
    by_type: dict[str, int] = {}
    for t in types.values():
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "by_type": by_type,
    }


def run_insights(
    df: pd.DataFrame, min_abs_corr: float = 0.5
) -> InsightsReport:
    """Run the full insights pipeline on a (cleaned) DataFrame."""
    types = infer_column_types(df)

    column_stats = compute_column_stats(df, types)
    numeric_cols = [c for c, t in types.items() if t in _NUMERIC_TYPES]
    correlations = compute_correlations(df, numeric_cols, min_abs_r=min_abs_corr)
    trends = compute_trends(df, types)
    category_breakdowns = compute_category_breakdowns(df, types)
    sentences = render_sentences(
        trends, correlations, category_breakdowns, column_stats
    )

    return InsightsReport(
        schema=_build_schema(df, types),
        columns=column_stats,
        correlations=correlations,
        trends=trends,
        category_breakdowns=category_breakdowns,
        insight_sentences=sentences,
    )
