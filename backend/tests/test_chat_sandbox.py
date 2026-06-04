"""Tests for app.chat.sandbox — AST validation + restricted exec + timeout."""

from __future__ import annotations

import pandas as pd

from app.chat.sandbox import (
    CodeTimeoutError,
    execute_code,
    validate_code,
)


def test_validate_clean_code_passes() -> None:
    assert validate_code("result = df.groupby('a')['b'].sum()") == []


def test_validate_blocks_import_os() -> None:
    violations = validate_code("import os\nresult = 1")
    assert any("os" in v for v in violations)


def test_validate_blocks_from_import() -> None:
    violations = validate_code("from os import path\nresult = 1")
    assert any("os" in v for v in violations)


def test_validate_blocks_subprocess() -> None:
    violations = validate_code("import subprocess")
    assert any("subprocess" in v for v in violations)


def test_validate_blocks_open_call() -> None:
    violations = validate_code("open('/etc/passwd').read()")
    assert any("open" in v for v in violations)


def test_validate_blocks_exec_call() -> None:
    violations = validate_code("exec('print(1)')")
    assert any("exec" in v for v in violations)


def test_validate_blocks_eval_call() -> None:
    violations = validate_code("eval('1+1')")
    assert any("eval" in v for v in violations)


def test_validate_blocks_dunder_access() -> None:
    violations = validate_code("df.__class__")
    assert any("__class__" in v for v in violations)


def test_validate_blocks_dunder_subclasses() -> None:
    violations = validate_code("().__class__.__bases__[0].__subclasses__()")
    assert any("dunder" in v.lower() or "__" in v for v in violations)


def test_validate_blocks_dunder_in_call() -> None:
    violations = validate_code("getattr(df, '__dict__')")
    assert any("__dict__" in v for v in violations)


def test_validate_syntax_error_returned() -> None:
    violations = validate_code("def x(:\n  pass")
    assert any("SyntaxError" in v for v in violations)


def test_execute_runs_simple_expression() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    val, err = execute_code("result = int(df['a'].sum())", df)
    assert err is None
    assert val == 6


def test_execute_runs_groupby() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    val, err = execute_code("result = df.groupby('g')['v'].sum()", df)
    assert err is None
    assert int(val.sum()) == 6


def test_execute_blocks_violation_via_validate() -> None:
    """run_code should refuse to exec code that has validation violations."""
    df = pd.DataFrame({"a": [1, 2]})
    # The executor itself runs validate_code first, so this returns an error
    # without ever calling exec.
    from app.chat.executor import run_code

    out = run_code("import os\nresult = 1", df)
    assert out["error"] is not None
    assert "os" in out["error"]
    assert out["result"]["type"] == "none"


def test_execute_swallows_runtime_error() -> None:
    df = pd.DataFrame({"a": [1]})
    val, err = execute_code("result = 1/0", df)
    assert val is None
    assert err is not None
    assert "ZeroDivisionError" in err


def test_execute_reports_name_error() -> None:
    df = pd.DataFrame({"a": [1]})
    val, err = execute_code("result = unknown_name", df)
    assert val is None
    assert err is not None
    assert "NameError" in err


def test_execute_blocks_open_in_safe_builtins() -> None:
    df = pd.DataFrame({"a": [1]})
    val, err = execute_code("result = open", df)
    # `open` was removed from safe builtins, so this raises NameError.
    assert val is None
    assert err is not None


def test_execute_blocks_eval_in_safe_builtins() -> None:
    df = pd.DataFrame({"a": [1]})
    val, err = execute_code("result = eval", df)
    assert val is None
    assert err is not None


def test_execute_timeout_triggers() -> None:
    """A busy loop with timeout=0 raises CodeTimeoutError and is caught."""
    df = pd.DataFrame({"a": [1]})
    # Use a short timeout; the loop should be killed.
    val, err = execute_code(
        "i = 0\nwhile True:\n    i += 1\nresult = i", df, timeout_s=1
    )
    # The handler should have fired, surfacing the timeout error.
    assert val is None
    assert err is not None
    assert "time limit" in err.lower() or "timeout" in err.lower()


def test_execute_handles_zero_timeout() -> None:
    """timeout_s=0 disables SIGALRM and runs the code normally."""
    df = pd.DataFrame({"a": [1]})
    val, err = execute_code("result = df['a'].iloc[0]", df, timeout_s=0)
    assert err is None
    assert int(val) == 1


def test_execute_can_use_safe_builtins() -> None:
    df = pd.DataFrame({"a": [3, 1, 2]})
    val, err = execute_code("result = sorted(df['a'].tolist())", df)
    assert err is None
    assert list(val) == [1, 2, 3]


def test_execute_provides_pd_and_np() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    val, err = execute_code("result = pd.DataFrame({'x': [10, 20]})['x'].sum()", df)
    assert err is None
    assert int(val) == 30


def test_validate_blocks_requests_import() -> None:
    violations = validate_code("import requests")
    assert any("requests" in v for v in violations)


def test_validate_blocks_socket_import() -> None:
    violations = validate_code("import socket")
    assert any("socket" in v for v in violations)


def test_validate_blocks_urllib_import() -> None:
    violations = validate_code("from urllib.request import urlopen")
    assert any("urllib" in v for v in violations)


def test_validate_blocks_asyncio_import() -> None:
    violations = validate_code("import asyncio")
    assert any("asyncio" in v for v in violations)


def test_validate_blocks_compile_call() -> None:
    violations = validate_code("compile('x = 1', '<s>', 'exec')")
    assert any("compile" in v for v in violations)


def test_validate_blocks_breakpoint_call() -> None:
    violations = validate_code("breakpoint()")
    assert any("breakpoint" in v for v in violations)


def test_validate_blocks_subprocess_attribute() -> None:
    # `subprocess` is a module name in FORBIDDEN_NAMES. Attribute access to
    # the name `subprocess.foo` should be caught as a forbidden name read.
    violations = validate_code("result = subprocess.call(['ls'])")
    # either the Name or the attribute rule will catch it
    assert len(violations) > 0


def test_execute_handles_legacy_codetimeout_marker() -> None:
    """The CodeTimeoutError symbol is importable from sandbox."""
    # Just make sure the symbol is exported and not accidentally renamed.
    assert CodeTimeoutError.__name__ == "CodeTimeoutError"
