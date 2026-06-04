"""Whitespace and column-name normalization for tabular data."""

from __future__ import annotations

import re

import pandas as pd


_SNAKE_RE = re.compile(r"[^0-9a-zA-Z]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _to_snake_case(name: str) -> str:
    name = name.strip()
    name = _CAMEL_BOUNDARY_RE.sub("_", name)
    name = _SNAKE_RE.sub("_", name)
    name = name.strip("_").lower()
    return name or "column"


def normalize_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Return a copy of ``df`` with snake_case, de-duplicated column names.

    Returns the new frame and a mapping ``original -> final`` so callers can
    surface renames in the cleaning report.
    """
    rename_map: dict[str, str] = {}
    seen: dict[str, int] = {}
    final_cols: list[str] = []
    for original in df.columns:
        base = _to_snake_case(str(original))
        count = seen.get(base, 0)
        candidate = base if count == 0 else f"{base}_{count}"
        seen[base] = count + 1
        final_cols.append(candidate)
        rename_map[str(original)] = candidate

    out = df.copy()
    out.columns = final_cols
    return out, rename_map


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading/trailing whitespace from every string-typed cell."""
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].apply(lambda v: v.strip() if isinstance(v, str) else v)
    return out
