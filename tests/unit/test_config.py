"""Smoke tests for config."""
from mlra.config import Settings


def test_default_model():
    s = Settings(groq_api_key="test-key")
    assert s.groq_model == "llama-3.3-70b-versatile"


def test_empty_key_allowed(monkeypatch, tmp_path):
    """No key set is fine — UI falls back to demo mode."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)  # avoid loading project's .env
    s = Settings()
    assert s.groq_api_key == ""
