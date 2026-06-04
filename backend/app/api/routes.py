"""HTTP routes for the cleaning + insights + dashboard + chat API."""

from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from ..chat.executor import run_code
from ..chat.llm import (
    LLMUnavailableError,
    build_messages,
    call_llm,
    code_and_answer,
)
from ..chat.schema import build_schema_prompt
from ..chat.session import chat_sessions
from ..cleaner.pipeline import run_cleaning
from ..cleaner.report import CleaningReport
from ..cleaner.type_detect import infer_column_type
from ..core.config import settings
from ..core.jobs import job_store
from ..dashboards.serialization import (
    dataframe_to_columns,
    dataframe_to_records,
    sample_dataframe,
)
from ..dashboards.suggestions import column_types_for, suggest_charts
from ..insights.pipeline import run_insights
from ..models.problem import ProblemInfo, detect_problem_type, is_supported
from ..models.result_store import ml_result_store
from ..models.train import (
    MIN_ROWS as _ML_MIN_ROWS,
    build_comparison_plotly_spec,
    train_and_evaluate,
)
from .schemas import (
    CategoryBreakdownInsightSchema,
    ChatRequest,
    ChatResponse,
    ChatResultSchema,
    ChartSpecSchema,
    CleanResponse,
    CleaningReportSchema,
    ColumnReportSchema,
    ColumnStatsSchema,
    CorrelationInsightSchema,
    DataResponse,
    InsightsReportSchema,
    InsightsResponse,
    MLFeatureSuggestion,
    MLModelResult,
    MLProblemInfo,
    MLReportResponse,
    MLSuggestRequest,
    MLSuggestResponse,
    MLTrainRequest,
    SchemaOverviewSchema,
    SuggestChartsResponse,
    TrendInsightSchema,
)


router = APIRouter()


def _read_upload(contents: bytes, filename: str) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(contents))
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(io.BytesIO(contents))
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file extension for '{filename}'. Allowed: {settings.allowed_extensions}",
    )


def _report_to_schema(report: CleaningReport) -> CleaningReportSchema:
    return CleaningReportSchema(
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        cols_in=report.cols_in,
        cols_out=report.cols_out,
        duplicates_removed=report.duplicates_removed,
        dropped_columns=list(report.dropped_columns),
        renamed_columns=dict(report.renamed_columns),
        columns=[
            ColumnReportSchema(
                name=c.name,
                detected_type=c.detected_type,
                nulls_before=c.nulls_before,
                nulls_after=c.nulls_after,
                action=c.action,
            )
            for c in report.columns
        ],
    )


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/clean", response_model=CleanResponse)
async def clean(
    file: UploadFile = File(...),
    formats: str = Query(
        default=",".join(settings.default_output_formats),
        description="Comma-separated list of output formats: csv,xlsx",
    ),
) -> CleanResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")

    if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension. Allowed: {settings.allowed_extensions}",
        )

    requested = [f.strip().lower() for f in formats.split(",") if f.strip()]
    for fmt in requested:
        if fmt not in ("csv", "xlsx"):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported output format '{fmt}'. Allowed: csv, xlsx",
            )

    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum size of {settings.max_upload_bytes} bytes",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        df = _read_upload(contents, file.filename)
    except Exception as exc:  # surface parse errors as 400
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows")

    cleaned_df, report = run_cleaning(df)
    job_id = job_store.create(cleaned_df, file.filename)

    downloads = {fmt: f"/download/{job_id}/{fmt}" for fmt in requested}
    return CleanResponse(
        job_id=job_id,
        report=_report_to_schema(report),
        downloads=downloads,
    )


@router.get("/download/{job_id}/{fmt}")
def download(job_id: str, fmt: str) -> StreamingResponse:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    if fmt == "csv":
        payload = job_store.serialize_csv(job)
        media_type = "text/csv"
        suffix = "csv"
    elif fmt == "xlsx":
        payload = job_store.serialize_xlsx(job)
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        suffix = "xlsx"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{fmt}'")

    base = job.original_filename.rsplit(".", 1)[0] or "cleaned"
    download_name = f"{base}_cleaned.{suffix}"

    return StreamingResponse(
        io.BytesIO(payload),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )


def _cleaning_report_to_schema(report: CleaningReport) -> CleaningReportSchema:
    return CleaningReportSchema(
        rows_in=report.rows_in,
        rows_out=report.rows_out,
        cols_in=report.cols_in,
        cols_out=report.cols_out,
        duplicates_removed=report.duplicates_removed,
        dropped_columns=list(report.dropped_columns),
        renamed_columns=dict(report.renamed_columns),
        columns=[
            ColumnReportSchema(
                name=c.name,
                detected_type=c.detected_type,
                nulls_before=c.nulls_before,
                nulls_after=c.nulls_after,
                action=c.action,
            )
            for c in report.columns
        ],
    )


def _insights_to_schema(report) -> InsightsReportSchema:
    return InsightsReportSchema(
        overview=SchemaOverviewSchema(**report.schema),
        columns={
            name: ColumnStatsSchema(type=payload["type"], stats=payload["stats"])
            for name, payload in report.columns.items()
        },
        correlations=[CorrelationInsightSchema(**c) for c in report.correlations],
        trends=[TrendInsightSchema(**t) for t in report.trends],
        category_breakdowns=[
            CategoryBreakdownInsightSchema(**cb) for cb in report.category_breakdowns
        ],
        insight_sentences=list(report.insight_sentences),
    )


async def _read_upload_file(
    file: UploadFile,
) -> tuple[pd.DataFrame, bytes]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file is missing a filename")
    if not any(file.filename.lower().endswith(ext) for ext in settings.allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension. Allowed: {settings.allowed_extensions}",
        )
    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Upload exceeds maximum size of {settings.max_upload_bytes} bytes",
        )
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        df = _read_upload(contents, file.filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file contains no rows")
    return df, contents


@router.post("/insights", response_model=InsightsResponse)
async def insights(
    file: UploadFile = File(...),
    min_corr: float = Query(default=0.5, ge=0.0, le=1.0),
) -> InsightsResponse:
    """Run cleaning + insights in one call. Returns both reports."""
    df, _ = await _read_upload_file(file)
    cleaned_df, cleaning_report = run_cleaning(df)
    job_id = job_store.create(cleaned_df, file.filename or "upload")
    insights_report = run_insights(cleaned_df, min_abs_corr=min_corr)
    return InsightsResponse(
        job_id=job_id,
        source="upload",
        cleaning_report=_cleaning_report_to_schema(cleaning_report),
        insights=_insights_to_schema(insights_report),
    )


@router.post("/insights/{job_id}", response_model=InsightsResponse)
def insights_from_job(
    job_id: str,
    min_corr: float = Query(default=0.5, ge=0.0, le=1.0),
) -> InsightsResponse:
    """Run insights against a previously cleaned job (no re-upload)."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    insights_report = run_insights(job.df, min_abs_corr=min_corr)
    return InsightsResponse(
        job_id=job_id,
        source=f"job:{job_id}",
        cleaning_report=None,
        insights=_insights_to_schema(insights_report),
    )


@router.get("/data/{job_id}", response_model=DataResponse)
def get_data(
    job_id: str,
    limit: int = Query(default=5000, ge=1, le=50000),
) -> DataResponse:
    """Return the cleaned DataFrame as JSON records, sampled if over the limit."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    total = len(job.df)
    df_to_send = sample_dataframe(job.df, limit=limit)
    return DataResponse(
        job_id=job_id,
        columns=dataframe_to_columns(df_to_send),
        records=dataframe_to_records(df_to_send),
        total_rows=total,
        returned_rows=len(df_to_send),
        sampled=len(df_to_send) < total,
    )


@router.post("/suggest-charts/{job_id}", response_model=SuggestChartsResponse)
def suggest_charts_for_job(
    job_id: str,
    max_charts: int = Query(default=8, ge=1, le=20, alias="max"),
) -> SuggestChartsResponse:
    """Return initial chart specs + column types for the dashboard."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    types = column_types_for(job.df)
    specs = suggest_charts(job.df, types=types, max_charts=max_charts)
    return SuggestChartsResponse(
        job_id=job_id,
        types=types,
        specs=[ChartSpecSchema(**spec) for spec in specs],
    )


@router.post("/chat/{job_id}", response_model=ChatResponse)
async def chat_with_job(
    job_id: str,
    request: ChatRequest,
) -> ChatResponse:
    """Ask the LLM a question about the cleaned DataFrame stored under ``job_id``."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    session_id = request.session_id or chat_sessions.new_session_id()
    history = chat_sessions.get_history(session_id)

    schema_desc = build_schema_prompt(job.df)
    messages = build_messages(schema_desc, history, request.message)

    try:
        parsed = await call_llm(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            messages=messages,
            timeout_s=settings.openai_timeout_s,
        )
    except LLMUnavailableError as exc:
        return ChatResponse(
            job_id=job_id,
            session_id=session_id,
            answer="",
            code="",
            result=ChatResultSchema(type="none"),
            plotly_spec=None,
            error=str(exc),
        )

    code, answer = code_and_answer(parsed)
    execution = run_code(code, job.df, timeout_s=settings.chat_code_timeout_s)

    response = ChatResponse(
        job_id=job_id,
        session_id=session_id,
        answer=answer,
        code=code,
        result=ChatResultSchema(**execution["result"]),
        plotly_spec=execution["plotly_spec"],
        error=execution["error"],
    )

    if execution["error"] is None:
        chat_sessions.add_exchange(session_id, request.message, answer)
    return response


# ---------------------------------------------------------------------------
# Phase 5: ML model suggestions
# ---------------------------------------------------------------------------


_USABLE_FEATURE_TYPES = {"integer", "float", "categorical", "boolean"}


def _pick_default_target(columns: list[str], types: dict[str, str]) -> str | None:
    """Heuristic default target: last non-text, non-datetime column."""
    for col in reversed(columns):
        if types.get(col) not in {"text", "datetime"}:
            return col
    return columns[-1] if columns else None


def _build_suggestion(
    job_id: str, target: str | None, df
) -> MLSuggestResponse | None:
    """Compute the auto-detected target, problem, and feature list."""
    types = column_types_for(df)
    if target is None:
        target = _pick_default_target(list(df.columns), types)
    if target is None or target not in df.columns:
        return None
    target_type = types.get(target, infer_column_type(df[target]))
    problem_info = detect_problem_type(df[target], detected_type=target_type)
    problem = MLProblemInfo(
        kind=problem_info.kind,
        reason=problem_info.reason,
        n_classes=problem_info.n_classes,
    )
    features = [
        MLFeatureSuggestion(name=c, detected_type=types.get(c, "text"), selected=True)
        for c in df.columns
        if c != target and types.get(c) in _USABLE_FEATURE_TYPES
    ]
    return MLSuggestResponse(
        job_id=job_id,
        target=target,
        target_type=target_type,
        problem=problem,
        suggested_features=features,
    )


@router.post("/ml/suggest/{job_id}", response_model=MLSuggestResponse)
def ml_suggest(job_id: str, request: MLSuggestRequest) -> MLSuggestResponse:
    """Suggest a target column, problem type, and feature list for ML training."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    target = request.target or None
    suggestion = _build_suggestion(job_id, target, job.df)
    if suggestion is None:
        raise HTTPException(
            status_code=400,
            detail="DataFrame has no columns that can be used as a target.",
        )
    return suggestion


@router.post("/ml/train/{job_id}", response_model=MLReportResponse)
def ml_train(job_id: str, request: MLTrainRequest) -> MLReportResponse:
    """Train all registered models against the chosen target + features."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    if request.target not in job.df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{request.target}' not in DataFrame.",
        )

    target_type = column_types_for(job.df).get(
        request.target, infer_column_type(job.df[request.target])
    )
    problem_info = detect_problem_type(job.df[request.target], detected_type=target_type)
    problem_schema = MLProblemInfo(
        kind=problem_info.kind,
        reason=problem_info.reason,
        n_classes=problem_info.n_classes,
    )

    if not is_supported(problem_info):
        return MLReportResponse(
            job_id=job_id,
            target=request.target,
            target_type=target_type,
            problem=problem_schema,
            features=list(request.features),
            rows_used=0,
            rows_total=len(job.df),
            downsampled=False,
            cached=False,
            models=[],
            plotly_spec=None,
            note=None,
            error=problem_info.reason,
        )

    report = train_and_evaluate(
        job.df,
        target=request.target,
        problem=ProblemInfo(
            kind=problem_info.kind,
            reason=problem_info.reason,
            n_classes=problem_info.n_classes,
        ),
        features=list(request.features),
    )

    if report.error is None:
        ml_result_store.put(job_id, request.target, report)

    return MLReportResponse(
        job_id=job_id,
        target=request.target,
        target_type=target_type,
        problem=problem_schema,
        features=report.features,
        rows_used=report.rows_used,
        rows_total=report.rows_total,
        downsampled=report.downsampled,
        cached=False,
        models=[
            MLModelResult(
                name=m.name,
                metrics=m.metrics,
                train_seconds=m.train_seconds,
                is_winner=m.is_winner,
                error=m.error,
            )
            for m in report.models
        ],
        plotly_spec=build_comparison_plotly_spec(report),
        note=report.note,
        error=report.error,
    )


@router.get("/ml/result/{job_id}", response_model=MLReportResponse)
def ml_get_result(
    job_id: str, target: str = Query(..., min_length=1)
) -> MLReportResponse:
    """Return the cached ML report for ``(job_id, target)`` if any."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    report = ml_result_store.get(job_id, target)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached ML result for job_id='{job_id}', target='{target}'.",
        )

    target_type = column_types_for(job.df).get(
        target, infer_column_type(job.df[target])
    )
    problem_info = report.problem
    return MLReportResponse(
        job_id=job_id,
        target=target,
        target_type=target_type,
        problem=MLProblemInfo(
            kind=problem_info.kind,
            reason=problem_info.reason,
            n_classes=problem_info.n_classes,
        ),
        features=report.features,
        rows_used=report.rows_used,
        rows_total=report.rows_total,
        downsampled=report.downsampled,
        cached=True,
        models=[
            MLModelResult(
                name=m.name,
                metrics=m.metrics,
                train_seconds=m.train_seconds,
                is_winner=m.is_winner,
                error=m.error,
            )
            for m in report.models
        ],
        plotly_spec=build_comparison_plotly_spec(report),
        note=report.note,
        error=report.error,
    )
