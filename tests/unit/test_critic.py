"""Tests for the critic — diagnosis logic only."""
from mlra.agent.critic import diagnose_failure, stream_fix
from mlra.config import settings


def test_diagnose_clean_run_returns_none():
    """Successful run with all SUPPORTED verdicts → no failure."""
    assert diagnose_failure(0, ["RESULT: ok"], ["SUPPORTED"]) is None


def test_diagnose_nonzero_exit():
    msg = diagnose_failure(1, ["something", "ERROR: broken"], [])
    assert msg is not None
    assert "non-zero" in msg


def test_diagnose_error_in_stdout():
    msg = diagnose_failure(0, ["loading", "ERROR: division by zero"], [])
    assert msg is not None
    assert "ERROR" in msg


def test_diagnose_all_plots_contradicted():
    msg = diagnose_failure(0, ["RESULT: x=5"], ["CONTRADICTED", "CONTRADICTED"])
    assert msg is not None
    assert "CONTRADICTED" in msg


def test_diagnose_one_unclear_no_failure():
    """A single UNCLEAR plot is not enough to trigger the critic."""
    assert diagnose_failure(0, ["RESULT: ok"], ["UNCLEAR", "SUPPORTED"]) is None


def test_stream_fix_no_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    text = "".join(stream_fix("q", "plan", "code", "error"))
    assert "disabled" in text or "GROQ" in text
