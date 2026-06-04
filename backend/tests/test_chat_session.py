"""Tests for app.chat.session — in-memory chat session store."""

from __future__ import annotations

import threading

from app.chat.session import ChatSessionStore


def test_get_history_starts_empty() -> None:
    store = ChatSessionStore()
    assert store.get_history("missing") == []


def test_add_exchange_appends_pair() -> None:
    store = ChatSessionStore()
    sid = "s1"
    store.add_exchange(sid, "hi", "hello")
    history = store.get_history(sid)
    assert history == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]


def test_add_exchange_caps_window() -> None:
    store = ChatSessionStore(max_history=4)
    sid = "s1"
    for i in range(5):
        store.add_exchange(sid, f"u{i}", f"a{i}")
    history = store.get_history(sid)
    assert len(history) == 4
    # Last exchange's two messages are preserved.
    assert history[-2]["content"] == "u4"
    assert history[-1]["content"] == "a4"
    # And the oldest user message in the window is the one from exchange 3.
    assert history[0]["content"] == "u3"


def test_clear_removes_session() -> None:
    store = ChatSessionStore()
    sid = "s1"
    store.add_exchange(sid, "hi", "hello")
    store.clear(sid)
    assert store.get_history(sid) == []


def test_set_history_truncates() -> None:
    store = ChatSessionStore(max_history=2)
    store.set_history(
        "s1",
        [
            {"role": "user", "content": "u0"},
            {"role": "assistant", "content": "a0"},
            {"role": "user", "content": "u1"},
            {"role": "assistant", "content": "a1"},
        ],
    )
    history = store.get_history("s1")
    assert len(history) == 2
    assert history[0]["content"] == "u1"


def test_new_session_id_is_unique() -> None:
    store = ChatSessionStore()
    a = store.new_session_id()
    b = store.new_session_id()
    assert a != b
    assert len(a) > 8


def test_sessions_isolated_between_ids() -> None:
    store = ChatSessionStore()
    store.add_exchange("a", "msg", "reply")
    store.add_exchange("b", "msg", "reply2")
    assert store.get_history("a")[1]["content"] == "reply"
    assert store.get_history("b")[1]["content"] == "reply2"


def test_thread_safety() -> None:
    """Concurrent add_exchange calls do not lose messages or corrupt the list."""
    store = ChatSessionStore(max_history=10000)
    sid = "s1"
    n = 200

    def worker() -> None:
        for i in range(n):
            store.add_exchange(sid, "u", "a")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    history = store.get_history(sid)
    assert len(history) == 4 * n * 2
    # The window cap should have kicked in.
    assert len(history) <= 10000
