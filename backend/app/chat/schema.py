"""Build a compact schema description for the LLM prompt."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from ..cleaner.type_detect import infer_column_types


def _column_info(name: str, col_type: str, series: pd.Series) -> str:
    info = f"  {name} ({col_type})"
    non_null = series.dropna()
    if col_type in {"integer", "float"}:
        numeric = pd.to_numeric(non_null, errors="coerce").dropna()
        if not numeric.empty:
            info += f", range {numeric.min():.4g}..{numeric.max():.4g}"
    elif col_type == "datetime":
        coerced = pd.to_datetime(non_null, errors="coerce", format="mixed").dropna()
        if not coerced.empty:
            info += f", range {coerced.min().date()}..{coerced.max().date()}"
    elif col_type in {"categorical", "boolean"}:
        if not non_null.empty:
            unique = [str(v) for v in non_null.unique()[:8]]
            info += f", values: {unique}"
    else:
        # text
        if not non_null.empty:
            sample = str(non_null.iloc[0])
            if len(sample) > 30:
                sample = sample[:30] + "..."
            info += f", sample: '{sample}'"
    return info


def build_schema_prompt(
    df: pd.DataFrame, types: dict[str, str] | None = None, sample_rows: int = 3
) -> str:
    """Return a compact, LLM-friendly description of ``df``."""
    if types is None:
        types = infer_column_types(df)

    column_lines = [
        _column_info(col, types.get(col, "text"), df[col]) for col in df.columns
    ]
    columns_block = "\n".join(column_lines) if column_lines else "  (no columns)"

    sample_records: list[dict[str, Any]] = []
    for _, row in df.head(sample_rows).iterrows():
        record: dict[str, Any] = {}
        for col in df.columns:
            v = row[col]
            if hasattr(v, "isoformat"):
                v = v.isoformat()
            elif hasattr(v, "item"):
                v = v.item()
            record[col] = v
        sample_records.append(record)
    sample_block = json.dumps(sample_records, default=str, indent=2)

    row_count = len(df)
    return (
        f"The DataFrame has {row_count} rows and {len(df.columns)} columns.\n"
        f"Columns:\n{columns_block}\n\n"
        f"Sample rows ({sample_rows}):\n{sample_block}"
    )
