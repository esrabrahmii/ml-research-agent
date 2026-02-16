"""Tests for plot verifier — parsing logic only (no network call)."""
from mlra.agent.verifier import _parse, extract_result_line, verify
from mlra.config import settings


def test_parse_supported():
    text = (
        "Verdict: SUPPORTED\n"
        "Visual: Accuracy climbs from 0.91 at n=10 to 0.97 at n=100, then plateaus.\n"
        "Note: Matches the claim that more data helps until ~100 samples."
    )
    r = _parse(text)
    assert r.verdict == "SUPPORTED"
    assert "Accuracy climbs" in r.visual
    assert "matches" in r.note.lower()


def test_parse_contradicted():
    text = "Verdict: CONTRADICTED\nVisual: Flat line.\nNote: No improvement visible."
    r = _parse(text)
    assert r.verdict == "CONTRADICTED"


def test_parse_unclear_default():
    """Garbage input should default to UNCLEAR."""
    r = _parse("blah blah whatever")
    assert r.verdict == "UNCLEAR"


def test_extract_result_line():
    log = ["loading...", "step 1", "RESULT: accuracy 0.97 at n=100", "done"]
    assert "RESULT" in extract_result_line(log)


def test_extract_result_line_fallback():
    log = ["just some output", "no result line", "really nothing"]
    assert extract_result_line(log) != ""


def test_verify_no_key_returns_unclear(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "google_api_key", "")
    fake_plot = tmp_path / "x.png"
    fake_plot.write_bytes(b"\x89PNG\r\n")
    r = verify(fake_plot, "RESULT: anything")
    assert r.verdict == "UNCLEAR"
