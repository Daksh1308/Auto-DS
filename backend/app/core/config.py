"""Application configuration."""

from __future__ import annotations

import os


class Settings:
    max_upload_bytes: int = int(os.environ.get("ADS_MAX_UPLOAD_BYTES", 50 * 1024 * 1024))
    allowed_extensions: tuple[str, ...] = (".csv", ".xlsx", ".xls")
    default_output_formats: tuple[str, ...] = ("csv", "xlsx")

    # LLM (OpenAI-compatible) — used by /chat/{job_id}
    openai_api_key: str | None = os.environ.get("OPENAI_API_KEY") or None
    openai_base_url: str = os.environ.get(
        "OPENAI_BASE_URL", "https://api.groq.com/openai/v1"
    )
    openai_model: str = os.environ.get("OPENAI_MODEL", "llama-3.3-70b-versatile")
    openai_timeout_s: int = int(os.environ.get("OPENAI_TIMEOUT_S", "30"))
    chat_code_timeout_s: int = int(os.environ.get("ADS_CHAT_CODE_TIMEOUT_S", "10"))


settings = Settings()
