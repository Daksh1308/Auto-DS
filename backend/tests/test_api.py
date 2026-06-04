"""End-to-end test against the FastAPI app using TestClient."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_healthz() -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_clean_csv_uploads_and_reports(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/clean",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
        )
    assert r.status_code == 200, r.text
    body = r.json()
    assert "job_id" in body
    assert set(body["downloads"].keys()) == {"csv", "xlsx"}
    report = body["report"]
    assert report["rows_in"] == 12
    assert report["duplicates_removed"] == 1
    assert "mostly_empty" in report["dropped_columns"]


def test_download_csv_returns_cleaned_file(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        post = client.post(
            "/clean",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
            params={"formats": "csv"},
        )
    job_id = post.json()["job_id"]
    dl = client.get(f"/download/{job_id}/csv")
    assert dl.status_code == 200
    assert dl.headers["content-type"].startswith("text/csv")
    cleaned = pd.read_csv(io.BytesIO(dl.content))
    assert cleaned.isna().sum().sum() == 0
    assert all(c.islower() and " " not in c for c in cleaned.columns)


def test_download_xlsx_returns_workbook(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        post = client.post(
            "/clean",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
            params={"formats": "xlsx"},
        )
    job_id = post.json()["job_id"]
    dl = client.get(f"/download/{job_id}/xlsx")
    assert dl.status_code == 200
    assert "spreadsheetml" in dl.headers["content-type"]
    cleaned = pd.read_excel(io.BytesIO(dl.content), sheet_name="cleaned")
    assert cleaned.isna().sum().sum() == 0


def test_clean_xlsx_round_trip(dirty_xlsx_path: Path) -> None:
    with dirty_xlsx_path.open("rb") as fh:
        r = client.post(
            "/clean",
            files={
                "file": (
                    dirty_xlsx_path.name,
                    fh,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
            params={"formats": "csv,xlsx"},
        )
    assert r.status_code == 200
    assert set(r.json()["downloads"].keys()) == {"csv", "xlsx"}


def test_rejects_unsupported_extension() -> None:
    r = client.post(
        "/clean",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
    assert "Unsupported" in r.json()["detail"]


def test_rejects_bad_format_param(dirty_csv_path: Path) -> None:
    with dirty_csv_path.open("rb") as fh:
        r = client.post(
            "/clean",
            files={"file": (dirty_csv_path.name, fh, "text/csv")},
            params={"formats": "xml"},
        )
    assert r.status_code == 400
    assert "xml" in r.json()["detail"]


def test_404_on_unknown_job() -> None:
    r = client.get("/download/does-not-exist/csv")
    assert r.status_code == 404
