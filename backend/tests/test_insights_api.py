"""End-to-end tests for the /insights endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_insights_from_upload(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/insights",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "upload"
    assert "job_id" in body
    assert body["cleaning_report"] is not None
    insights = body["insights"]
    # /insights runs on the CLEANED frame: 12 input - 1 duplicate = 11 rows
    assert insights["overview"]["rows"] == 11
    assert "insight_sentences" in insights
    assert isinstance(insights["insight_sentences"], list)


def test_insights_from_job_id(dirty_csv_path: Path) -> None:
    # First, run /clean to obtain a job_id.
    with dirty_csv_path.open("rb") as fh:
        clean = client.post(
            "/clean",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
        )
    job_id = clean.json()["job_id"]

    # Re-use that job for insights; no re-upload.
    r = client.post(f"/insights/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["source"] == f"job:{job_id}"
    # Re-use path doesn't re-run the cleaner; no cleaning_report.
    assert body["cleaning_report"] is None


def test_insights_min_corr_query_param(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/insights",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
            params={"min_corr": 0.99},
        )
    assert r.status_code == 200
    body = r.json()
    # At min_corr=0.99, almost nothing should pass; empty list is fine.
    assert body["insights"]["correlations"] == []


def test_insights_404_for_unknown_job() -> None:
    r = client.post("/insights/does-not-exist")
    assert r.status_code == 404


def test_insights_rejects_unsupported_extension() -> None:
    r = client.post(
        "/insights",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400


def test_insights_response_shape_matches_schema(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/insights",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
        )
    insights = r.json()["insights"]
    # Top-level keys
    for key in (
        "overview",
        "columns",
        "correlations",
        "trends",
        "category_breakdowns",
        "insight_sentences",
    ):
        assert key in insights
    # Each column entry has a type and stats dict
    for name, payload in insights["columns"].items():
        assert "type" in payload
        assert "stats" in payload
        assert isinstance(name, str)


def test_insights_downloaded_csv_is_the_cleaned_df(dirty_csv_path: Path) -> None:
    """End-to-end: upload to /insights, then download the cleaned CSV from the
    returned job_id and confirm the data is the same as what insights ran on."""
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/insights",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
        )
    job_id = r.json()["job_id"]
    dl = client.get(f"/download/{job_id}/csv")
    assert dl.status_code == 200
    df = pd.read_csv(io.BytesIO(dl.content))
    # The dirty fixture has 12 rows, 1 duplicate, 1 high-null column dropped -> 11 rows, 6 cols
    assert len(df) == 11
    assert df.shape[1] == 6
    assert df.isna().sum().sum() == 0
