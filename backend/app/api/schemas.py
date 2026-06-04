"""Pydantic schemas for the cleaning + insights + dashboard + chat API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


OutputFormat = Literal["csv", "xlsx"]
ChartType = Literal["line", "bar", "pie"]
Aggregation = Literal["sum", "mean", "count", "median", "min", "max"]
ColumnType = Literal[
    "integer", "float", "datetime", "boolean", "categorical", "text"
]


class ColumnReportSchema(BaseModel):
    name: str
    detected_type: str
    nulls_before: int
    nulls_after: int
    action: str


class CleaningReportSchema(BaseModel):
    rows_in: int
    rows_out: int
    cols_in: int
    cols_out: int
    duplicates_removed: int
    dropped_columns: list[str] = Field(default_factory=list)
    renamed_columns: dict[str, str] = Field(default_factory=dict)
    columns: list[ColumnReportSchema] = Field(default_factory=list)


class CleanResponse(BaseModel):
    job_id: str
    report: CleaningReportSchema
    downloads: dict[str, str]


class SchemaOverviewSchema(BaseModel):
    rows: int
    cols: int
    by_type: dict[str, int] = Field(default_factory=dict)


class ColumnStatsSchema(BaseModel):
    type: str
    stats: dict = Field(default_factory=dict)


class CorrelationInsightSchema(BaseModel):
    a: str
    b: str
    r: float


class TrendInsightSchema(BaseModel):
    column: str
    period_col: str
    first_period: str
    last_period: str
    first_value: float
    last_value: float
    pct_change: float
    direction: str


class CategoryBreakdownInsightSchema(BaseModel):
    category_col: str
    numeric_col: str
    top: dict
    bottom: dict
    pct_diff: float


class InsightsReportSchema(BaseModel):
    overview: SchemaOverviewSchema
    columns: dict[str, ColumnStatsSchema] = Field(default_factory=dict)
    correlations: list[CorrelationInsightSchema] = Field(default_factory=list)
    trends: list[TrendInsightSchema] = Field(default_factory=list)
    category_breakdowns: list[CategoryBreakdownInsightSchema] = Field(default_factory=list)
    insight_sentences: list[str] = Field(default_factory=list)


class InsightsResponse(BaseModel):
    job_id: str
    source: str
    cleaning_report: CleaningReportSchema | None = None
    insights: InsightsReportSchema


class ChartSpecSchema(BaseModel):
    id: str
    type: ChartType
    title: str
    x_column: str
    y_column: str | None = None
    aggregation: Aggregation
    plotly_spec: dict = Field(default_factory=dict)


class SuggestChartsResponse(BaseModel):
    job_id: str
    types: dict[str, str] = Field(default_factory=dict)
    specs: list[ChartSpecSchema] = Field(default_factory=list)


class DataResponse(BaseModel):
    job_id: str
    columns: list[str] = Field(default_factory=list)
    records: list[dict] = Field(default_factory=list)
    total_rows: int
    returned_rows: int
    sampled: bool


ChatResultType = Literal["none", "dataframe", "scalar"]


class ChatRequest(BaseModel):
    session_id: str | None = Field(
        default=None,
        description="Opaque chat session id. A new one is generated if omitted.",
    )
    message: str = Field(..., min_length=1, description="User question for the LLM.")


class ChatResultSchema(BaseModel):
    type: ChatResultType
    value: object | None = None
    columns: list[str] = Field(default_factory=list)
    records: list[dict] = Field(default_factory=list)


class ChatResponse(BaseModel):
    job_id: str
    session_id: str
    answer: str
    code: str
    result: ChatResultSchema
    plotly_spec: dict | None = None
    error: str | None = None


# ----- Phase 5: ML model suggestions -----


MLProblemKind = Literal[
    "regression",
    "binary_classification",
    "multiclass_classification",
    "unsupported",
]


class MLProblemInfo(BaseModel):
    kind: MLProblemKind
    reason: str
    n_classes: int | None = None


class MLFeatureSuggestion(BaseModel):
    name: str
    detected_type: ColumnType
    selected: bool = True


class MLSuggestRequest(BaseModel):
    target: str | None = Field(
        default=None,
        description="Target column. If omitted, the last eligible column is used.",
    )


class MLSuggestResponse(BaseModel):
    job_id: str
    target: str
    target_type: ColumnType
    problem: MLProblemInfo
    suggested_features: list[MLFeatureSuggestion] = Field(default_factory=list)
    note: str | None = None


class MLTrainRequest(BaseModel):
    target: str = Field(..., min_length=1, description="Target column.")
    features: list[str] = Field(..., min_length=1, description="Feature columns.")


class MLModelResult(BaseModel):
    name: str
    metrics: dict[str, float] = Field(default_factory=dict)
    train_seconds: float
    is_winner: bool = False
    error: str | None = None


class MLReportResponse(BaseModel):
    job_id: str
    target: str
    target_type: ColumnType
    problem: MLProblemInfo
    features: list[str]
    rows_used: int
    rows_total: int
    downsampled: bool
    cached: bool
    models: list[MLModelResult] = Field(default_factory=list)
    plotly_spec: dict | None = None
    note: str | None = None
    error: str | None = None
