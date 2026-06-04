"""Runtime smoke test — patches the LLM call to return canned responses and
hits the real running uvicorn server with a real cleaned file."""

from __future__ import annotations

import json
import sys
import threading
import time
from typing import Any

import httpx
import pandas as pd
import uvicorn

sys.path.insert(0, ".")

from app.main import app  # noqa: E402
from app.chat.llm import call_llm  # noqa: E402


def _make_stub():
    async def _stub_call_llm(*args: Any, **kwargs: Any) -> dict[str, Any]:
        """Pretend to be the LLM — return a hard-coded but valid code+answer."""
        user_message = ""
        msgs = kwargs.get("messages") or (args[3] if len(args) >= 4 else None) or []
        if msgs:
            user_message = msgs[-1].get("content", "")
        lower = user_message.lower()
        if "os" in lower:
            return {"code": "import os\nresult = 1", "answer": "x"}
        if "math" in lower or "divide" in lower:
            return {"code": "result = 1/0", "answer": "x"}
        if "trend" in lower:
            return {
                "code": "result = df.groupby('sign_up_date')['age'].sum().reset_index()",
                "answer": "x",
            }
        if "mean" in lower:
            code = "result = float(df['age'].mean())"
            answer = "Mean age computed from the cleaned DataFrame."
        elif "city" in lower:
            code = "result = df.groupby('city')['age'].sum().reset_index()"
            answer = "Totals by city."
        else:
            code = "result = int(df['age'].sum())"
            answer = "Sum of age."

        return {"code": code, "answer": answer}

    return _stub_call_llm


_stub = _make_stub()

# Monkey-patch BEFORE the server starts handling requests.
import app.chat.llm as llm_mod  # noqa: E402

llm_mod.call_llm = _stub  # type: ignore[assignment]
# routes imports the symbol directly; rebind there too.
import app.api.routes as routes  # noqa: E402

routes.call_llm = _stub  # type: ignore[assignment]


def _run() -> None:
    config = uvicorn.Config(app, host="127.0.0.1", port=8765, log_level="warning")
    uvicorn.Server(config).run()


t = threading.Thread(target=_run, daemon=True)
t.start()
time.sleep(1.5)

base = "http://127.0.0.1:8765"
print("health:", httpx.get(f"{base}/healthz", timeout=2).json())

with open("tests/fixtures/dirty.csv", "rb") as f:
    r = httpx.post(
        f"{base}/clean", files={"file": ("dirty.csv", f, "text/csv")}, timeout=5
    )
r.raise_for_status()
job_id = r.json()["job_id"]
print("job:", job_id)

# 1. Scalar mean
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "what is the mean of age?"},
    timeout=5,
)
body = r.json()
print("MEAN:", json.dumps(body, indent=2))
assert body["error"] is None
assert body["result"]["type"] == "scalar"
assert isinstance(body["result"]["value"], float)

# 2. Dataframe groupby
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "totals by city", "session_id": "smoke-sess"},
    timeout=5,
)
body = r.json()
print("CITY:", json.dumps(body, indent=2))
assert body["error"] is None
assert body["result"]["type"] == "dataframe"
assert "city" in body["result"]["columns"]
assert body["plotly_spec"] is not None
assert "data" in body["plotly_spec"]
assert body["session_id"] == "smoke-sess"

# 3. Multi-turn — verify the history is sent back. We can't see the request,
#    but we can confirm the second turn with the same session also succeeds.
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "follow up", "session_id": "smoke-sess"},
    timeout=5,
)
body = r.json()
print("FOLLOWUP:", json.dumps(body, indent=2))
assert body["error"] is None

# 4. Sandbox violation surfaces
import app.chat.llm as llm_mod_again  # noqa: E402

async def _bad_code(*a: Any, **k: Any) -> dict[str, Any]:
    return {"code": "import os\nresult = 1", "answer": "x"}


llm_mod_again.call_llm = _bad_code  # type: ignore[assignment]
routes.call_llm = _bad_code  # type: ignore[assignment]
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "os version"},
    timeout=5,
)
body = r.json()
print("VIOLATION:", json.dumps(body, indent=2))
assert body["error"] is not None
assert "os" in body["error"]

# 5. Runtime error
async def _runtime_err(*a: Any, **k: Any) -> dict[str, Any]:
    return {"code": "result = 1/0", "answer": "x"}


llm_mod_again.call_llm = _runtime_err  # type: ignore[assignment]
routes.call_llm = _runtime_err  # type: ignore[assignment]
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "math"},
    timeout=5,
)
body = r.json()
print("RUNTIME:", json.dumps(body, indent=2))
assert body["error"] is not None
assert "ZeroDivisionError" in body["error"]

# 6. Auto-chart on a datetime + numeric table
async def _trend(*a: Any, **k: Any) -> dict[str, Any]:
    return {
        "code": "result = df.groupby('sign_up_date')['age'].sum().reset_index()",
        "answer": "x",
    }


llm_mod_again.call_llm = _trend  # type: ignore[assignment]
routes.call_llm = _trend  # type: ignore[assignment]
r = httpx.post(
    f"{base}/chat/{job_id}",
    json={"message": "trend"},
    timeout=5,
)
body = r.json()
print("DATETIME:", json.dumps(body, indent=2))
assert body["error"] is None
assert body["plotly_spec"] is not None

print("\nALL SMOKE CHECKS PASSED")
