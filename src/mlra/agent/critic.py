"""Critic / Reflection node — when an experiment fails or the plot verifier
disagrees with the agent's claim, the critic reads the failure + the code +
the plan, identifies the root cause, and proposes a corrected script.

We use the same Groq model with a different system prompt focused on
diagnosis-then-fix. The output is the fixed Python script (no commentary).
"""
from __future__ import annotations

from collections.abc import Iterator

from mlra.config import settings

CRITIC_SYSTEM_PROMPT = """You are a senior ML engineer doing emergency code review on a failed experiment.

You will be given:
1. The research question
2. The original plan
3. The Python code that was generated
4. The failure mode — either an error message, or a plot-verifier disagreement

Your job: identify the root cause and produce a CORRECTED version of the code.

STRICT RULES (same as the original code generator):
- Use ONLY: numpy, pandas, scikit-learn, matplotlib.
- Total runtime under 60 seconds.
- Save plots via `plt.savefig("plot_N.png", dpi=120, bbox_inches='tight')`. NEVER plt.show().
- Always StandardScaler before linear models, max_iter=5000 for LogisticRegression.
- Fixed test_size=0.2, random_state=42. Don't re-split inside loops.
- End with a single `RESULT: ...` line. Wrap in try/except with `ERROR:` on failure.
- Print at most ~15 lines of stdout.

ADDITIONAL RULE: address the SPECIFIC failure. Don't rewrite from scratch — minimal targeted fix.
If the plot verifier said the plot contradicts the claim, fix the BUG that caused the wrong claim
(usually a logic error in how the metric was computed or summarised).

Output ONLY the corrected Python code. No markdown fences. No commentary.
"""


def stream_fix(
    question: str,
    plan: str,
    code: str,
    failure: str,
) -> Iterator[str]:
    """Stream a corrected script."""
    if not settings.groq_api_key:
        yield "# (critic disabled — no GROQ_API_KEY)\n"
        return

    from groq import Groq

    user_msg = (
        f"# Research question\n{question}\n\n"
        f"# Plan\n{plan}\n\n"
        f"# Original code (failed)\n{code}\n\n"
        f"# Failure mode\n{failure}\n\n"
        f"Write the fixed code now."
    )

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=2000,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def diagnose_failure(return_code: int, stdout_lines: list[str], verdicts: list[str]) -> str | None:
    """Return a 1-paragraph failure description, or None if everything went fine."""
    last_lines = "\n".join(stdout_lines[-20:])
    if return_code != 0:
        return (
            f"The script exited with non-zero return code {return_code}. "
            f"Last 20 lines of stdout:\n{last_lines}"
        )
    if any("ERROR:" in line for line in stdout_lines):
        err_lines = [line for line in stdout_lines if "ERROR:" in line]
        return f"The script printed an ERROR line: {err_lines[0]}\n\nContext:\n{last_lines}"
    contradicted = sum(1 for v in verdicts if v == "CONTRADICTED")
    if contradicted and contradicted == len(verdicts):
        return (
            "The plot verifier flagged ALL plots as CONTRADICTED — "
            "the visual evidence does not support the claimed RESULT. "
            f"Stdout was:\n{last_lines}"
        )
    return None
