"""Reporter — generates a polished markdown research report after the
experiment runs, incorporating the agent's plan, code, stdout, the plot
verifier's independent analysis, and the arXiv references.

This is the *deliverable* the interviewer reads.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from mlra.config import settings


@dataclass
class ReportContext:
    question: str
    plan: str
    stdout: str
    verifier_verdicts: list[str]    # ["SUPPORTED", "UNCLEAR", ...]
    verifier_visuals: list[str]     # short descriptions
    references: list[str]           # ["Smith et al. 2024 — arXiv:2401.00001", ...]
    n_plots: int
    n_retries: int                  # how many times the critic kicked in


REPORT_SYSTEM_PROMPT = """You are a precise research scientist writing a one-page report on an ML experiment that just finished.

Render a markdown report with these EXACT sections:

## TL;DR
One or two sentences with the headline finding. Lead with the number.

## Method
3-4 sentences describing what was tested and how.

## Results
A markdown table if there are numbers; otherwise prose. Include concrete numbers from stdout.

## Verification
1-2 sentences. Did the plot verifier (independent vision model) support, contradict, or find unclear the agent's claim? Quote what it saw.

## Limitations
2-3 short bullets — sample size, single seed, scope, etc.

## References
List the arXiv references provided. Format: - [Smith et al. 2024](url) — short title

Be terse and professional. Total ~300 words.
Do NOT invent numbers — only use what's in the stdout.
Do NOT invent references — only use what's provided.
"""


def stream_report(ctx: ReportContext) -> Iterator[str]:
    """Stream the markdown report token-by-token."""
    if not settings.groq_api_key:
        yield "# Final report disabled — no GROQ_API_KEY"
        return

    from groq import Groq

    verdict_summary = ", ".join(ctx.verifier_verdicts) if ctx.verifier_verdicts else "(no verifier run)"
    visual_summary = "\n".join(f"- Plot {i+1}: {v}" for i, v in enumerate(ctx.verifier_visuals)) or "(no visuals)"
    refs = "\n".join(f"- {r}" for r in ctx.references) or "(no references)"

    user_msg = (
        f"# Question\n{ctx.question}\n\n"
        f"# Plan\n{ctx.plan}\n\n"
        f"# Last 30 lines of stdout\n{ctx.stdout[-3000:]}\n\n"
        f"# Plot verifier verdicts ({ctx.n_plots} plot(s))\n"
        f"Verdicts: {verdict_summary}\n"
        f"{visual_summary}\n\n"
        f"# Reflection retries\nThe critic kicked in {ctx.n_retries} time(s).\n\n"
        f"# Available references\n{refs}\n\n"
        f"Write the report now."
    )

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=1200,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
