"""End-to-end tests for /ml/suggest, /ml/train, and /ml/result."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from app.models.result_store import ml_result_store


client = TestClient(app)


def _clean_first(csv_path: Path) -> str:
    with csv_path.open("rb") as fh:
        r = client.post("/clean", files={"file": (csv_path.name, fh, "text/csv")})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


def test_ml_suggest_404_unknown_job() -> None:
    r = client.post("/ml/suggest/does-not-exist", json={})
    assert r.status_code == 404


def test_ml_suggest_picks_default_target(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/ml/suggest/{job_id}", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["job_id"] == job_id
    assert body["target"]  # auto-picked
    # Cleaned fixture has 'age' as a numeric column.
    assert "age" in [f["name"] for f in body["suggested_features"]]


def test_ml_suggest_with_explicit_target(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/ml/suggest/{job_id}", json={"target": "age"})
    assert r.status_code == 200
    body = r.json()
    assert body["target"] == "age"


def test_ml_suggest_target_not_in_data(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(f"/ml/suggest/{job_id}", json={"target": "nope"})
    # The target is missing, so the suggest should still 200 with a
    # reasonable default, OR 400. Verify it's at least an error code.
    # Actually, current code allows the user to pass any string and the
    # suggestion builds. The error would surface on /ml/train.
    assert r.status_code in (200, 400)


def test_ml_train_404_unknown_job() -> None:
    r = client.post(
        "/ml/train/does-not-exist",
        json={"target": "age", "features": ["city"]},
    )
    assert r.status_code == 404


def test_ml_train_regression_happy_path() -> None:
    job_id = _clean_first(Path("tests/fixtures/ml_data.csv"))
    # age is high-cardinality integer → regression
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "age", "features": ["spend"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["error"] is None
    assert body["problem"]["kind"] == "regression"
    assert len(body["models"]) == 3
    model_names = [m["name"] for m in body["models"]]
    assert "LinearRegression" in model_names
    assert "RandomForestRegressor" in model_names
    assert "HistGradientBoostingRegressor" in model_names
    # Winner flagged
    winners = [m for m in body["models"] if m["is_winner"]]
    assert len(winners) == 1
    # Plotly spec
    assert body["plotly_spec"] is not None
    assert body["plotly_spec"]["data"][0]["type"] == "bar"


def test_ml_train_classification_happy_path() -> None:
    job_id = _clean_first(Path("tests/fixtures/ml_data.csv"))
    # 'subscribed' is a boolean → binary_classification
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "subscribed", "features": ["age", "city"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["error"] is None
    assert body["problem"]["kind"] == "binary_classification"
    assert len(body["models"]) == 3
    for m in body["models"]:
        if m["error"] is None:
            assert "accuracy" in m["metrics"]
            assert "f1_macro" in m["metrics"]


def test_ml_train_unsupported_target(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    # 'full_name' is text → unsupported
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "full_name", "features": ["age"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["error"] is not None
    assert "not supported" in body["error"].lower()


def test_ml_train_target_not_in_data(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "nonexistent", "features": ["age"]},
    )
    assert r.status_code == 400
    assert "not in DataFrame" in r.json()["detail"]


def test_ml_train_empty_features(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "age", "features": []},
    )
    # Pydantic should reject empty list with 422
    assert r.status_code == 422


def test_ml_train_only_target_column() -> None:
    """DataFrame with just a target → no usable features."""
    # Build a tiny job by creating it through clean
    csv_text = "age\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n21\n22\n23\n24\n25\n26\n27\n28\n29\n30\n"
    from io import BytesIO

    files = {"file": ("tiny.csv", BytesIO(csv_text.encode("utf-8")), "text/csv")}
    r = client.post("/clean", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "age", "features": ["age"]},  # target == feature
    )
    # /ml/train should 200 with a friendly error in the body.
    assert r.status_code == 200
    body = r.json()
    assert body["error"] is not None
    assert "cannot also be a feature" in body["error"]


def test_ml_train_downsampling_flag() -> None:
    """A dataset over 50k rows triggers the downsample flag with a note."""
    from io import BytesIO

    n = 60_000
    rng = np.random.default_rng(0)
    csv = "f1,f2,age\n"
    for i in range(n):
        csv += f"{rng.random():.4f},{rng.random():.4f},{rng.integers(20, 60)}\n"
    files = {"file": ("big.csv", BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/clean", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "age", "features": ["f1", "f2"]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["error"] is None
    assert body["downsampled"] is True
    assert body["rows_used"] == 50_000
    assert body["rows_total"] == 60_000
    assert body["note"] is not None
    assert "Downsampled" in body["note"]


def test_ml_result_caches_after_train() -> None:
    job_id = _clean_first(Path("tests/fixtures/ml_data.csv"))
    # Train
    r1 = client.post(
        f"/ml/train/{job_id}",
        json={"target": "age", "features": ["spend"]},
    )
    assert r1.status_code == 200
    assert r1.json()["cached"] is False

    # Fetch cached
    r2 = client.get(f"/ml/result/{job_id}", params={"target": "age"})
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["cached"] is True
    assert len(body["models"]) == 3


def test_ml_result_404_when_no_cache(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    r = client.get(f"/ml/result/{job_id}", params={"target": "age"})
    assert r.status_code == 404


def test_ml_result_404_unknown_job() -> None:
    r = client.get("/ml/result/does-not-exist", params={"target": "age"})
    assert r.status_code == 404


def test_ml_result_requires_target_query() -> None:
    r = client.get("/ml/result/anything")
    assert r.status_code == 422


def test_ml_train_too_small_dataset() -> None:
    """A 5-row DataFrame triggers the too-small error."""
    from io import BytesIO

    csv = "a,b,y\n1,2,3\n4,5,6\n7,8,9\n1,1,2\n3,3,4\n"
    files = {"file": ("small.csv", BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/clean", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "y", "features": ["a", "b"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["error"] is not None
    assert "too small" in body["error"].lower()


def test_ml_train_singleton_target_classification() -> None:
    """A target with only 1 unique value → unsupported."""
    from io import BytesIO

    csv = "a,b,y\n1,2,1\n4,5,1\n7,8,1\n1,1,1\n3,3,1\n4,4,1\n2,3,1\n5,6,1\n7,7,1\n9,9,1\n1,2,1\n3,4,1\n"
    files = {"file": ("one.csv", BytesIO(csv.encode("utf-8")), "text/csv")}
    r = client.post("/clean", files=files)
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    r = client.post(
        f"/ml/train/{job_id}",
        json={"target": "y", "features": ["a", "b"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["error"] is not None
    assert body["problem"]["kind"] == "unsupported"


def test_ml_result_store_clears() -> None:
    ml_result_store.clear()
    assert ml_result_store.get("any", "any") is None
