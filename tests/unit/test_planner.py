"""Tests for the planner. The demo branch runs without any API key."""
from mlra.agent.planner import _demo_plan, stream_plan
from mlra.config import settings


def test_demo_plan_yields_text():
    text = "".join(_demo_plan())
    assert "Hypothesis" in text
    assert "Method" in text
    assert "Estimated time" in text


def test_stream_plan_demo_mode_when_no_key(monkeypatch):
    """When no API key is set, stream_plan should use the demo branch."""
    monkeypatch.setattr(settings, "groq_api_key", "")
    text = "".join(stream_plan("does anything matter?"))
    assert len(text) > 100
    assert "Hypothesis" in text and "Method" in text
