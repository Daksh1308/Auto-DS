"""Run LLM-generated code on the cleaned DataFrame and format the result."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ..dashboards.plotly_specs import build_plotly_spec
from .sandbox import execute_code, validate_code


MAX_RESULT_ROWS = 1000


def _safe_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        return pd.Timestamp(value).isoformat()
    return value


def _dataframe_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    capped = df.head(MAX_RESULT_ROWS)
    out: list[dict[str, Any]] = []
    for _, row in capped.iterrows():
        out.append({col: _to_jsonable(row[col]) for col in capped.columns})
    return out


def _maybe_build_chart(df: pd.DataFrame) -> dict | None:
    """Auto-build a Plotly figure when the result has 1 numeric + 1 other column.

    Returns ``None`` for any other shape; the frontend just shows the table.
    """
    if df.empty or len(df.columns) < 2:
        return None
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) != 1 or len(df.columns) > 2:
        return None
    y_col = numeric_cols[0]
    x_col = next(c for c in df.columns if c != y_col)

    is_datetime = pd.api.types.is_datetime64_any_dtype(df[x_col])
    chart_type = "line" if is_datetime else "bar"
    title = f"{y_col} by {x_col}"

    try:
        return build_plotly_spec(
            chart_type=chart_type,
            df=df,
            x_column=x_col,
            y_column=y_col,
            aggregation="sum",
            title=title,
        )
    except Exception:
        return None


def _empty_result() -> dict:
    return {
        "type": "none",
        "value": None,
        "columns": [],
        "records": [],
    }


def _format_result(value: Any) -> dict[str, Any]:
    if value is None:
        return {
            "result": _empty_result(),
            "plotly_spec": None,
            "error": None,
        }

    if isinstance(value, pd.DataFrame):
        return {
            "result": {
                "type": "dataframe",
                "value": None,
                "columns": [str(c) for c in value.columns],
                "records": _dataframe_to_records(value),
            },
            "plotly_spec": _maybe_build_chart(value),
            "error": None,
        }

    if isinstance(value, pd.Series):
        df = value.reset_index()
        df.columns = [str(c) for c in df.columns]
        return {
            "result": {
                "type": "dataframe",
                "value": None,
                "columns": list(df.columns),
                "records": _dataframe_to_records(df),
            },
            "plotly_spec": _maybe_build_chart(df),
            "error": None,
        }

    if isinstance(value, (bool, np.bool_)):
        return {
            "result": {"type": "scalar", "value": bool(value), "columns": [], "records": []},
            "plotly_spec": None,
            "error": None,
        }
    if isinstance(value, (int, np.integer)):
        return {
            "result": {"type": "scalar", "value": int(value), "columns": [], "records": []},
            "plotly_spec": None,
            "error": None,
        }
    if isinstance(value, (float, np.floating)):
        safe = _safe_float(value)
        return {
            "result": {"type": "scalar", "value": safe, "columns": [], "records": []},
            "plotly_spec": None,
            "error": None,
        }
    if isinstance(value, str):
        return {
            "result": {"type": "scalar", "value": value, "columns": [], "records": []},
            "plotly_spec": None,
            "error": None,
        }

    # Anything else — coerce to string
    return {
        "result": {"type": "scalar", "value": str(value), "columns": [], "records": []},
        "plotly_spec": None,
        "error": None,
    }


def run_code(code: str, df: pd.DataFrame, timeout_s: int = 10) -> dict[str, Any]:
    """Validate, execute, and format the result of LLM-generated code.

    Always returns a dict with keys ``result``, ``plotly_spec``, and ``error``.
    """
    violations = validate_code(code)
    if violations:
        return {
            "result": _empty_result(),
            "plotly_spec": None,
            "error": "; ".join(violations),
        }

    value, exec_error = execute_code(code, df, timeout_s=timeout_s)
    if exec_error is not None:
        return {
            "result": _empty_result(),
            "plotly_spec": None,
            "error": exec_error,
        }
    return _format_result(value)
