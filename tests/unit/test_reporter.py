"""Tests for the report writer."""
from mlra.agent.reporter import ReportContext, stream_report
from mlra.config import settings


def test_report_context_dataclass():
    ctx = ReportContext(
        question="q",
        plan="p",
        stdout="RESULT: 0.97",
        verifier_verdicts=["SUPPORTED"],
        verifier_visuals=["accuracy curve climbs"],
        references=["Smith et al. 2024 — https://arxiv.org/abs/2401.00001"],
        n_plots=1,
        n_retries=0,
    )
    assert ctx.question == "q"
    assert ctx.n_plots == 1


def test_stream_report_no_key(monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "")
    ctx = ReportContext(
        question="q", plan="p", stdout="", verifier_verdicts=[],
        verifier_visuals=[], references=[], n_plots=0, n_retries=0,
    )
    text = "".join(stream_report(ctx))
    assert "disabled" in text or "GROQ" in text
