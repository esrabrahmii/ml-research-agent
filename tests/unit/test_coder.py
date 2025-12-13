"""Tests for the coder. Demo branch runs without an API key."""
from mlra.agent.coder import _demo_code, stream_code
from mlra.config import settings


def test_demo_code_yields_python():
    text = "".join(_demo_code())
    assert "import" in text
    assert "matplotlib" in text or "sklearn" in text
    assert "plt.savefig" in text or "savefig" in text


def test_stream_code_demo_when_no_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    text = "".join(stream_code("any q", "any plan"))
    assert len(text) > 100
    assert "import" in text
