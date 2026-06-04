"""OpenAI-compatible LLM client + JSON-mode prompt assembly."""

from __future__ import annotations

import json
from typing import Any

import httpx

from .schema import build_schema_prompt


SYSTEM_PROMPT_TEMPLATE = """You are a data analyst assistant. The user has a pandas DataFrame named `df`.

{schema}

You can use the variables: df (the cleaned DataFrame), pd (pandas), np (numpy).
Write Python code that uses pandas to answer the question, and assign the final result to a variable named `result`.
- For tabular results, set `result` to a pandas DataFrame or Series.
- For a single value, set `result` to that scalar.
- For "why" questions, set `result` to the supporting data and put your explanation in the `answer` field.
- Do NOT import any modules; only use pd, np, and df.

Respond with strict JSON in this exact format:
{{"code": "<python code>", "answer": "<one-sentence natural-language answer>"}}

Do not include any text outside the JSON object.
"""


class LLMUnavailableError(RuntimeError):
    """Raised when the LLM cannot be called (no key, bad config, network error)."""


def build_messages(
    schema_description: str, history: list[dict], user_message: str
) -> list[dict[str, str]]:
    system = SYSTEM_PROMPT_TEMPLATE.format(schema=schema_description)
    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


async def call_llm(
    api_key: str | None,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat completions endpoint and return the parsed JSON.

    Raises ``LLMUnavailableError`` for transport / parsing failures.
    """
    if not api_key:
        raise LLMUnavailableError("No LLM API key configured (set OPENAI_API_KEY)")

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            res = await client.post(url, json=payload, headers=headers)
    except httpx.HTTPError as e:
        raise LLMUnavailableError(f"LLM transport error: {e}") from e

    if res.status_code >= 400:
        # Truncate the body — providers can return huge error pages.
        body = (res.text or "")[:500]
        raise LLMUnavailableError(
            f"LLM returned HTTP {res.status_code}: {body}"
        )

    try:
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (ValueError, KeyError, IndexError) as e:
        raise LLMUnavailableError(f"LLM response was not valid JSON: {e}") from e

    if not isinstance(parsed, dict):
        raise LLMUnavailableError("LLM response was not a JSON object")
    return parsed


def code_and_answer(parsed: dict[str, Any]) -> tuple[str, str]:
    code = parsed.get("code")
    answer = parsed.get("answer")
    if not isinstance(code, str):
        code = ""
    if not isinstance(answer, str):
        answer = ""
    return code, answer
