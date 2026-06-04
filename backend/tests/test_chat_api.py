"""End-to-end tests for POST /chat/{job_id}."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
from fastapi.testclient import TestClient

from app.chat.session import chat_sessions
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


class _StubTransport(httpx.AsyncBaseTransport):
    """httpx stub that returns a fixed JSON LLM response."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.calls = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return self._response


def _ok_chat_completion(code: str, answer: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"code": code, "answer": answer}),
                    }
                }
            ]
        },
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def _patch_llm_transport(response: httpx.Response) -> _StubTransport:
    """Monkeypatch httpx.AsyncClient so it routes through our stub."""
    import app.chat.llm as llm_mod

    transport = _StubTransport(response)
    orig = httpx.AsyncClient

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return orig(*args, **kwargs)

    llm_mod.httpx.AsyncClient = _factory  # type: ignore[assignment]
    # Stash for cleanup
    llm_mod._orig_async_client = orig  # type: ignore[attr-defined]
    return transport


def _restore_llm_transport() -> None:
    import app.chat.llm as llm_mod

    llm_mod.httpx.AsyncClient = llm_mod._orig_async_client  # type: ignore[attr-defined]


def _patch_settings(api_key: str = "k", base_url: str = "https://api.groq.com/openai/v1") -> None:
    import app.api.routes as routes
    from app.core import config as cfg

    real_settings = routes.settings
    real_settings.openai_api_key = api_key
    real_settings.openai_base_url = base_url
    real_settings.openai_model = "test-model"
    real_settings.openai_timeout_s = 5
    real_settings.chat_code_timeout_s = 5
    cfg.settings = real_settings  # in case anything imports cfg.settings


def test_chat_404_unknown_job() -> None:
    r = client.post(
        "/chat/does-not-exist",
        json={"message": "what is the mean of a?"},
    )
    assert r.status_code == 404


def test_chat_requires_message() -> None:
    job_id = _clean_first(Path("tests/fixtures/dirty.csv"))
    r = client.post(f"/chat/{job_id}", json={})
    assert r.status_code == 422


def test_chat_503_when_no_api_key(dirty_csv_path: Path) -> None:
    """Without a configured key, the route returns a 200 with a friendly error.

    The frontend can show the error message inline; an LLM outage is not a
    server bug, so 503 isn't appropriate.
    """
    # Save and clear key
    from app.core import config as cfg

    real_settings = cfg.settings
    saved = real_settings.openai_api_key
    real_settings.openai_api_key = None

    try:
        # Need a job
        job_id = _clean_first(dirty_csv_path)
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "what is the mean of age?"},
        )
    finally:
        real_settings.openai_api_key = saved

    assert r.status_code == 200
    body = r.json()
    assert body["error"] is not None
    assert "OPENAI_API_KEY" in body["error"]


def test_chat_happy_path_returns_code_and_result(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    code = "result = int(df['age'].sum())"
    _patch_settings()
    transport = _patch_llm_transport(_ok_chat_completion(code, "Sum is computed."))
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "what is the total?", "session_id": "test-s1"},
        )
    finally:
        _restore_llm_transport()
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] == "test-s1"
    assert body["code"] == code
    assert body["answer"] == "Sum is computed."
    assert body["error"] is None
    assert body["result"]["type"] == "scalar"
    # Real value from the dirty.csv fixture (post-cleaning); LLM stub returns code.
    assert body["result"]["value"] == 374
    assert transport.calls == 1


def test_chat_dataframe_result_and_chart(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    code = "result = df.groupby('city')['age'].sum().reset_index()"
    _patch_settings()
    _patch_llm_transport(_ok_chat_completion(code, "Grouped totals."))
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "totals by city"},
        )
    finally:
        _restore_llm_transport()
    body = r.json()
    assert body["error"] is None
    assert body["result"]["type"] == "dataframe"
    assert "city" in body["result"]["columns"]
    assert body["plotly_spec"] is not None
    assert "data" in body["plotly_spec"]


def test_chat_invalid_code_returns_error(dirty_csv_path: Path) -> None:
    """Code that violates the sandbox must surface a 200 with an error string."""
    job_id = _clean_first(dirty_csv_path)
    bad_code = "import os\nresult = 1"
    _patch_settings()
    _patch_llm_transport(_ok_chat_completion(bad_code, ""))
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "os version"},
        )
    finally:
        _restore_llm_transport()
    body = r.json()
    assert body["error"] is not None
    assert "os" in body["error"]
    assert body["result"]["type"] == "none"


def test_chat_runtime_error_returns_error(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    code = "result = 1/0"
    _patch_settings()
    _patch_llm_transport(_ok_chat_completion(code, ""))
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "divide by zero"},
        )
    finally:
        _restore_llm_transport()
    body = r.json()
    assert body["error"] is not None
    assert "ZeroDivisionError" in body["error"]


def test_chat_session_memory_accumulates(dirty_csv_path: Path) -> None:
    """Two requests with the same session_id should pass accumulated history."""
    job_id = _clean_first(dirty_csv_path)
    _patch_settings()
    transport = _patch_llm_transport(
        _ok_chat_completion("result = 1", "ok")
    )
    try:
        # First turn
        r1 = client.post(
            f"/chat/{job_id}",
            json={"message": "first", "session_id": "sess-A"},
        )
        # Second turn
        r2 = client.post(
            f"/chat/{job_id}",
            json={"message": "second", "session_id": "sess-A"},
        )
    finally:
        _restore_llm_transport()
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json()["session_id"] == "sess-A"
    assert r2.json()["session_id"] == "sess-A"
    assert transport.calls == 2

    history = chat_sessions.get_history("sess-A")
    assert len(history) == 4  # 2 user + 2 assistant
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_chat_session_memory_not_persisted_on_error(dirty_csv_path: Path) -> None:
    """Failed turns should not pollute the history."""
    job_id = _clean_first(dirty_csv_path)
    _patch_settings()
    _patch_llm_transport(
        _ok_chat_completion("import os\nresult = 1", "")
    )
    sid = "sess-err"
    try:
        client.post(
            f"/chat/{job_id}",
            json={"message": "anything", "session_id": sid},
        )
    finally:
        _restore_llm_transport()
    assert chat_sessions.get_history(sid) == []


def test_chat_generates_session_id_when_missing(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    _patch_settings()
    _patch_llm_transport(_ok_chat_completion("result = 1", "ok"))
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "hello"},
        )
    finally:
        _restore_llm_transport()
    body = r.json()
    assert body["session_id"]
    assert len(body["session_id"]) > 8


def test_chat_llm_transport_error_surfaces(dirty_csv_path: Path) -> None:
    job_id = _clean_first(dirty_csv_path)
    _patch_settings()
    _patch_llm_transport(
        httpx.Response(
            500,
            text="boom",
            request=httpx.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            ),
        )
    )
    try:
        r = client.post(
            f"/chat/{job_id}",
            json={"message": "hello"},
        )
    finally:
        _restore_llm_transport()
    body = r.json()
    assert body["error"] is not None
    assert "HTTP 500" in body["error"]
