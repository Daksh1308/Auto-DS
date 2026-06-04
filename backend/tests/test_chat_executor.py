"""Tests for app.chat.executor — end-to-end run_code behavior."""

from __future__ import annotations

import pandas as pd

from app.chat.executor import run_code


def test_run_code_scalar_int() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    out = run_code("result = int(df['a'].sum())", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["result"]["value"] == 6


def test_run_code_scalar_float() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = run_code("result = float(df['a'].mean())", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["result"]["value"] == 1.5


def test_run_code_scalar_string() -> None:
    df = pd.DataFrame({"a": ["hello"]})
    out = run_code("result = df['a'].iloc[0]", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["result"]["value"] == "hello"


def test_run_code_scalar_bool() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    out = run_code("result = (df['a'].sum() > 0)", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["result"]["value"] is True


def test_run_code_dataframe_result() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    out = run_code("result = df.groupby('g')['v'].sum().reset_index()", df)
    assert out["error"] is None
    assert out["result"]["type"] == "dataframe"
    assert "g" in out["result"]["columns"]
    assert "v" in out["result"]["columns"]
    assert len(out["result"]["records"]) == 2


def test_run_code_series_result_is_formatted_as_dataframe() -> None:
    df = pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]})
    out = run_code("result = df.groupby('g')['v'].sum()", df)
    assert out["error"] is None
    assert out["result"]["type"] == "dataframe"
    # Two columns: original groupby index + 'v'
    assert len(out["result"]["columns"]) == 2
    assert len(out["result"]["records"]) == 2


def test_run_code_auto_builds_chart_for_one_numeric_column() -> None:
    df = pd.DataFrame({"g": ["a", "b", "c"], "v": [10, 20, 30]})
    out = run_code("result = df.groupby('g')['v'].sum().reset_index()", df)
    assert out["error"] is None
    assert out["plotly_spec"] is not None
    assert "data" in out["plotly_spec"]
    assert "layout" in out["plotly_spec"]


def test_run_code_auto_builds_line_chart_for_datetime_x() -> None:
    df = pd.DataFrame(
        {
            "d": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "v": [1, 2, 3],
        }
    )
    out = run_code("result = df.copy()", df)
    assert out["error"] is None
    assert out["plotly_spec"] is not None
    trace = out["plotly_spec"]["data"][0]
    assert trace["type"] == "scatter"


def test_run_code_no_chart_for_non_tabular() -> None:
    df = pd.DataFrame({"a": [1, 2]})
    out = run_code("result = 42", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["plotly_spec"] is None


def test_run_code_no_chart_for_multi_numeric_columns() -> None:
    df = pd.DataFrame({"g": ["a", "b"], "x": [1, 2], "y": [3, 4]})
    out = run_code("result = df.copy()", df)
    assert out["plotly_spec"] is None


def test_run_code_violation_returns_error() -> None:
    df = pd.DataFrame({"a": [1]})
    out = run_code("import os\nresult = 1", df)
    assert out["error"] is not None
    assert out["result"]["type"] == "none"
    assert out["plotly_spec"] is None


def test_run_code_runtime_error_returns_error() -> None:
    df = pd.DataFrame({"a": [1]})
    out = run_code("result = 1/0", df)
    assert out["error"] is not None
    assert "ZeroDivisionError" in out["error"]


def test_run_code_caps_records_at_1000() -> None:
    df = pd.DataFrame({"i": list(range(2000)), "v": list(range(2000))})
    out = run_code("result = df", df)
    assert out["error"] is None
    assert out["result"]["type"] == "dataframe"
    assert len(out["result"]["records"]) == 1000


def test_run_code_handles_nan_in_records() -> None:
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = run_code("result = df", df)
    assert out["error"] is None
    # The NaN in row 1 should be serialized as null
    for v in out["result"]["records"][0].values():
        assert v is None or isinstance(v, (str, int, float, bool))


def test_run_code_dataframe_with_only_text_columns_no_chart() -> None:
    df = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    out = run_code("result = df", df)
    assert out["error"] is None
    assert out["plotly_spec"] is None


def test_run_code_scalar_none_serialized() -> None:
    df = pd.DataFrame({"a": [1]})
    out = run_code("result = None", df)
    assert out["error"] is None
    # A None result intentionally maps to type="none" so the frontend can
    # hide the result panel.
    assert out["result"]["type"] == "none"


def test_run_code_handles_unexpected_value_type() -> None:
    df = pd.DataFrame({"a": [1]})
    # set, list, dict, tuple are not in the explicit type branch — they fall
    # through to the str() coercion path. None of these should crash.
    out = run_code("result = [1, 2, 3]", df)
    assert out["error"] is None
    assert out["result"]["type"] == "scalar"
    assert out["result"]["value"] == "[1, 2, 3]"
