"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="session")
def dirty_csv_path() -> Path:
    return FIXTURES_DIR / "dirty.csv"


@pytest.fixture(scope="session")
def dirty_xlsx_path() -> Path:
    return FIXTURES_DIR / "dirty.xlsx"


@pytest.fixture(scope="session")
def dirty_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "dirty.csv")
