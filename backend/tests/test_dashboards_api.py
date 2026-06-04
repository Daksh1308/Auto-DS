"""End-to-end tests for the dashboard API endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _clean_first(csv_path: Path) -> str:
    with csv_path.open("rb") as fh:
        r = client.post(
            "/clean",
            files={"file": (csv_path.name, fh, "text/csv")},
        )
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_get_data_returns_records_and_columns(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.get(f"/data/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert "customer_id" in body["columns"]
    assert isinstance(body["records"], list)
    assert body["total_rows"] == 11
    assert body["returned_rows"] == 11
    assert body["sampled"] is False


def test_get_data_sampling(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.get(f"/data/{job_id}", params={"limit": 5})
    body = r.json()
    assert body["returned_rows"] == 5
    assert body["sampled"] is True
    assert body["total_rows"] == 11


def test_get_data_404_unknown_job() -> None:
    r = client.get("/data/does-not-exist")
    assert r.status_code == 404


def test_get_data_records_are_jsonable(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.get(f"/data/{job_id}")
    records = r.json()["records"]
    # Every record value must be JSON-serializable (no NaN/Infinity in output).
    for record in records:
        for value in record.values():
            assert value is None or isinstance(value, (str, int, float, bool))


def test_suggest_charts_returns_specs_and_types(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/suggest-charts/{job_id}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert isinstance(body["types"], dict)
    assert "datetime" in body["types"].values()
    assert isinstance(body["specs"], list)
    assert len(body["specs"]) > 0
    for spec in body["specs"]:
        assert spec["type"] in {"line", "bar", "pie"}
        assert "plotly_spec" in spec


def test_suggest_charts_respects_max(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/suggest-charts/{job_id}", params={"max": 2})
    body = r.json()
    assert len(body["specs"]) <= 2


def test_suggest_charts_404_unknown_job() -> None:
    r = client.post("/suggest-charts/does-not-exist")
    assert r.status_code == 404


def test_suggest_charts_plotly_specs_have_data_and_layout(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/suggest-charts/{job_id}")
    for spec in r.json()["specs"]:
        plotly = spec["plotly_spec"]
        assert "data" in plotly
        assert "layout" in plotly
        assert "title" in plotly["layout"]
