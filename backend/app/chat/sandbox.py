"""AST validation + restricted exec + timeout for LLM-generated code."""

from __future__ import annotations

import ast
import builtins
import signal
from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_NAMES: set[str] = {
    "open",
    "exec",
    "eval",
    "compile",
    "globals",
    "locals",
    "__import__",
    "input",
    "breakpoint",
    "memoryview",
    "help",
    "dir",
    "vars",
}

FORBIDDEN_MODULES: set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "urllib",
    "http",
    "asyncio",
    "shutil",
    "pathlib",
    "tempfile",
    "ctypes",
    "multiprocessing",
    "threading",
}

DUNDER_ATTRS: set[str] = {
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "__code__",
    "__dict__",
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__import__",
    "__builtins__",
    "__init_subclass__",
    "__subclasshook__",
}


def validate_code(code: str) -> list[str]:
    """Return a list of violation messages; empty list means the code is safe."""
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e.msg}"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in FORBIDDEN_MODULES:
                    violations.append(f"Import of '{alias.name}' is not allowed")
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in FORBIDDEN_MODULES:
                violations.append(
                    f"Import from '{node.module or '?'}' is not allowed"
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in FORBIDDEN_NAMES:
                violations.append(f"Call to '{func.id}' is not allowed")
            # getattr/setattr/delattr with a string dunder -> flag the dunder.
            if isinstance(func, ast.Name) and func.id in {"getattr", "setattr", "delattr"}:
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                    dunder = node.args[1].value
                    if isinstance(dunder, str) and dunder in DUNDER_ATTRS:
                        violations.append(
                            f"{func.id} on dunder '{dunder}' is not allowed"
                        )
        elif isinstance(node, ast.Attribute):
            if node.attr in DUNDER_ATTRS:
                violations.append(f"Access to dunder '{node.attr}' is not allowed")
            if node.attr in FORBIDDEN_NAMES:
                violations.append(f"Attribute access to '{node.attr}' is not allowed")
        elif isinstance(node, ast.Name):
            if node.id in FORBIDDEN_MODULES:
                # Reading a bare module name like `subprocess`, `os`, `sys`
                # is almost always an attempt to escape the sandbox.
                violations.append(f"Reference to module '{node.id}' is not allowed")
            elif node.id in FORBIDDEN_NAMES and isinstance(node.ctx, (ast.Store, ast.Del)):
                violations.append(f"Use of '{node.id}' is not allowed")

    return violations


# A minimal, safe builtins dict.
SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name)
    for name in (
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "filter",
        "float",
        "frozenset",
        "int",
        "isinstance",
        "issubclass",
        "len",
        "list",
        "map",
        "max",
        "min",
        "print",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "slice",
        "sorted",
        "str",
        "sum",
        "tuple",
        "type",
        "zip",
        "True",
        "False",
        "None",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError",
        "StopIteration",
        "ZeroDivisionError",
        "ArithmeticError",
        "AttributeError",
        "NameError",
    )
}


class CodeTimeoutError(Exception):
    """Raised when sandboxed code exceeds the time limit."""


def _timeout_handler(signum: int, frame: Any) -> None:
    raise CodeTimeoutError("Code execution exceeded the time limit")


def execute_code(
    code: str, df: pd.DataFrame, timeout_s: int = 10
) -> tuple[Any, str | None]:
    """Execute ``code`` in a restricted env. Returns ``(result_value, error_message)``."""
    local_scope: dict[str, Any] = {}
    global_scope: dict[str, Any] = {
        "__builtins__": SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
    }

    old_handler = None
    timed = False
    if timeout_s and timeout_s > 0:
        try:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_s)
            timed = True
        except (AttributeError, ValueError):
            # Not on Unix, or called from a non-main thread.
            timed = False

    try:
        exec(code, global_scope, local_scope)
        return local_scope.get("result"), None
    except CodeTimeoutError:
        return None, f"Code execution exceeded the {timeout_s}s time limit"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        if timed and old_handler is not None:
            try:
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
            except (AttributeError, ValueError):
                pass
