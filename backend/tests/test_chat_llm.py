"""Tests for app.chat.llm — message building + LLM transport (httpx mocked)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.chat.llm import (
    LLMUnavailableError,
    SYSTEM_PROMPT_TEMPLATE,
    build_messages,
    call_llm,
    code_and_answer,
)


def test_build_messages_includes_system_user_and_history() -> None:
    history = [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
    ]
    msgs = build_messages("SCHEMA", history, "new question")
    assert msgs[0]["role"] == "system"
    assert "SCHEMA" in msgs[0]["content"]
    # history preserved in order, user message last
    assert msgs[1:] == [
        {"role": "user", "content": "earlier q"},
        {"role": "assistant", "content": "earlier a"},
        {"role": "user", "content": "new question"},
    ]


def test_system_prompt_template_contains_schema() -> None:
    rendered = SYSTEM_PROMPT_TEMPLATE.format(schema="COLUMNS: x (integer)")
    assert "COLUMNS: x (integer)" in rendered
    assert "df" in rendered
    assert "result" in rendered


def test_code_and_answer_handles_missing_fields() -> None:
    code, answer = code_and_answer({})
    assert code == ""
    assert answer == ""
    code, answer = code_and_answer({"code": "x = 1"})
    assert code == "x = 1"
    assert answer == ""
    code, answer = code_and_answer({"code": "x = 1", "answer": "done"})
    assert code == "x = 1"
    assert answer == "done"


def test_code_and_answer_coerces_non_string() -> None:
    code, answer = code_and_answer({"code": 123, "answer": ["list", "is", "bad"]})
    assert code == ""
    assert answer == ""


class _StubTransport(httpx.AsyncBaseTransport):
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self._response


def _ok_response(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
    )


def _with_patched_client(transport: _StubTransport):
    """Return a context manager that patches httpx.AsyncClient to use ``transport``."""
    import app.chat.llm as llm_mod

    orig = httpx.AsyncClient
    captured: list[httpx.AsyncClient] = []

    def _factory(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        c = orig(*args, **kwargs)
        captured.append(c)
        return c

    llm_mod.httpx.AsyncClient = _factory  # type: ignore[assignment]

    class _Ctx:
        def __enter__(self) -> _StubTransport:
            return transport

        def __exit__(self, *args: Any) -> None:
            llm_mod.httpx.AsyncClient = orig  # type: ignore[assignment]
            for c in captured:
                try:
                    c.close()
                except Exception:
                    pass

    return _Ctx()


def test_call_llm_sends_correct_request_and_parses_json() -> None:
    payload = {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"code": "result = 1", "answer": "one"}),
                }
            }
        ]
    }
    transport = _StubTransport(_ok_response(payload))
    with _with_patched_client(transport):
        parsed = asyncio.run(
            call_llm(
                api_key="k",
                base_url="https://api.groq.com/openai/v1",
                model="m",
                messages=[{"role": "user", "content": "q"}],
            )
        )
    assert parsed == {"code": "result = 1", "answer": "one"}
    assert len(transport.calls) == 1
    req = transport.calls[0]
    assert "Bearer k" in req.headers["authorization"]
    body = json.loads(req.content.decode("utf-8"))
    assert body["model"] == "m"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.0


def test_call_llm_raises_on_no_key() -> None:
    with _with_patched_client(_StubTransport(_ok_response({}))):
        try:
            asyncio.run(
                call_llm(
                    api_key=None,
                    base_url="https://api.groq.com/openai/v1",
                    model="m",
                    messages=[],
                )
            )
        except LLMUnavailableError as exc_info:
            assert "OPENAI_API_KEY" in str(exc_info)
            return
    raise AssertionError("expected LLMUnavailableError")


def test_call_llm_raises_on_http_error() -> None:
    transport = _StubTransport(
        httpx.Response(
            500,
            text="upstream down",
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )
    )
    with _with_patched_client(transport):
        try:
            asyncio.run(
                call_llm(
                    api_key="k",
                    base_url="https://api.groq.com/openai/v1",
                    model="m",
                    messages=[],
                )
            )
        except LLMUnavailableError as exc_info:
            assert "HTTP 500" in str(exc_info)
            return
    raise AssertionError("expected LLMUnavailableError")


def test_call_llm_raises_on_malformed_payload() -> None:
    payload = {"choices": []}  # missing [0]
    transport = _StubTransport(_ok_response(payload))
    with _with_patched_client(transport):
        try:
            asyncio.run(
                call_llm(
                    api_key="k",
                    base_url="https://api.groq.com/openai/v1",
                    model="m",
                    messages=[],
                )
            )
        except LLMUnavailableError:
            return
    raise AssertionError("expected LLMUnavailableError")


def test_call_llm_raises_on_non_object_json() -> None:
    payload = {
        "choices": [
            {"message": {"content": json.dumps(["not", "an", "object"])}}
        ]
    }
    transport = _StubTransport(_ok_response(payload))
    with _with_patched_client(transport):
        try:
            asyncio.run(
                call_llm(
                    api_key="k",
                    base_url="https://api.groq.com/openai/v1",
                    model="m",
                    messages=[],
                )
            )
        except LLMUnavailableError:
            return
    raise AssertionError("expected LLMUnavailableError")
