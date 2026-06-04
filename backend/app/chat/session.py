"""In-memory chat session store with a sliding window of recent messages."""

from __future__ import annotations

import uuid
from threading import Lock
from typing import TypedDict


class Message(TypedDict):
    role: str  # "user" | "assistant"
    content: str


class ChatSessionStore:
    def __init__(self, max_history: int = 6) -> None:
        self._sessions: dict[str, list[Message]] = {}
        self._lock = Lock()
        self._max = max_history

    def get_history(self, session_id: str) -> list[Message]:
        with self._lock:
            return list(self._sessions.get(session_id, []))

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, [])
            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})
            if len(history) > self._max:
                self._sessions[session_id] = history[-self._max :]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def set_history(self, session_id: str, messages: list[Message]) -> None:
        with self._lock:
            self._sessions[session_id] = messages[-self._max :]

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex


chat_sessions = ChatSessionStore()
