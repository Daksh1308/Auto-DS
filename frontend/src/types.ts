/** Shared types matching the backend pydantic schemas. */

export type ChartType = "line" | "bar" | "pie";
export type Aggregation = "sum" | "mean" | "count" | "median" | "min" | "max";
export type ColumnType =
  | "integer"
  | "float"
  | "datetime"
  | "boolean"
  | "categorical"
  | "text";

export interface ChartSpec {
  id: string;
  type: ChartType;
  title: string;
  x_column: string;
  y_column: string | null;
  aggregation: Aggregation;
  plotly_spec: PlotlyFigure;
}

export interface PlotlyFigure {
  data: Array<Record<string, unknown>>;
  layout: Record<string, unknown>;
}

export interface CleaningReport {
  rows_in: number;
  rows_out: number;
  cols_in: number;
  cols_out: number;
  duplicates_removed: number;
  dropped_columns: string[];
  renamed_columns: Record<string, string>;
  columns: Array<{
    name: string;
    detected_type: ColumnType;
    nulls_before: number;
    nulls_after: number;
    action: string;
  }>;
}

export interface CleanResponse {
  job_id: string;
  report: CleaningReport;
  downloads: Record<string, string>;
}

export interface DataResponse {
  job_id: string;
  columns: string[];
  records: Array<Record<string, unknown>>;
  total_rows: number;
  returned_rows: number;
  sampled: boolean;
}

export interface SuggestChartsResponse {
  job_id: string;
  types: Record<string, ColumnType>;
  specs: ChartSpec[];
}

export type ChatResultType = "none" | "dataframe" | "scalar";

export interface ChatResult {
  type: ChatResultType;
  value: unknown;
  columns: string[];
  records: Array<Record<string, unknown>>;
}

export interface ChatRequest {
  session_id?: string | null;
  message: string;
}

export interface ChatResponse {
  job_id: string;
  session_id: string;
  answer: string;
  code: string;
  result: ChatResult;
  plotly_spec: PlotlyFigure | null;
  error: string | null;
}

// ----- Phase 5: ML model suggestions -----

export type MLProblemKind =
  | "regression"
  | "binary_classification"
  | "multiclass_classification"
  | "unsupported";

export interface MLProblemInfo {
  kind: MLProblemKind;
  reason: string;
  n_classes: number | null;
}

export interface MLFeatureSuggestion {
  name: string;
  detected_type: ColumnType;
  selected: boolean;
}

export interface MLSuggestResponse {
  job_id: string;
  target: string;
  target_type: ColumnType;
  problem: MLProblemInfo;
  suggested_features: MLFeatureSuggestion[];
  note: string | null;
}

export interface MLModelResult {
  name: string;
  metrics: Record<string, number>;
  train_seconds: number;
  is_winner: boolean;
  error: string | null;
}

export interface MLReportResponse {
  job_id: string;
  target: string;
  target_type: ColumnType;
  problem: MLProblemInfo;
  features: string[];
  rows_used: number;
  rows_total: number;
  downsampled: boolean;
  cached: boolean;
  models: MLModelResult[];
  plotly_spec: PlotlyFigure | null;
  note: string | null;
  error: string | null;
}
