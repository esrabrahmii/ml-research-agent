"""Tests for the local subprocess executor."""
from mlra.agent.executor import execute, strip_code_fences


def test_strip_code_fences():
    assert strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert strip_code_fences("print(1)") == "print(1)"
    assert strip_code_fences("```\nx = 2\n```") == "x = 2"


def test_execute_runs_simple_code():
    code = "print('hello'); print('world')"
    events = list(execute(code, timeout=15))
    kinds = [k for k, _ in events]
    assert "run_id" in kinds
    assert "stdout" in kinds
    assert "done" in kinds
    stdout_lines = [v for k, v in events if k == "stdout"]
    assert "hello" in stdout_lines
    assert "world" in stdout_lines
    done = [v for k, v in events if k == "done"][0]
    assert done.return_code == 0


def test_execute_captures_errors():
    code = "raise ValueError('oops')"
    events = list(execute(code, timeout=10))
    done = [v for k, v in events if k == "done"][0]
    assert done.return_code != 0
