"""Plan generator — streams a step-by-step ML experimental plan from Groq.

This is the first node of the agent. Later weeks add: arXiv RAG before planning,
an executor that actually runs the plan, a critic that reflects, and a final
report writer.
"""
from __future__ import annotations

from collections.abc import Iterator

from mlra.config import settings

SYSTEM_PROMPT = """You are an expert ML researcher. Given a research question, produce a precise, ready-to-run experimental plan.

Format your response in this EXACT structure (use markdown):

**Hypothesis**: One sentence stating what we expect to find.

**Method** (3–7 numbered steps):
1. [Concrete step naming the exact model, dataset, and hyperparameters]
2. ...

**Metrics**: What we'll measure (with units).

**Estimated time**: ~X minutes on a Mac (no GPU).

Be concrete. Use specific HuggingFace model IDs (e.g. `EleutherAI/pythia-160m`), specific dataset splits (e.g. `gsm8k:main:train[:100]`), and exact hyperparameters. Keep the model size small enough to run on a laptop CPU in under 10 minutes.
"""


def stream_plan(question: str, literature: str | None = None) -> Iterator[str]:
    """Stream a research plan token-by-token.

    If `literature` is given (formatted arXiv excerpts), the planner is asked to
    ground its method in those references and cite them inline as [1], [2], etc.

    Falls back to a hard-coded demo plan if no GROQ_API_KEY is set, so the UI
    works on first launch without configuration.
    """
    if not settings.groq_api_key:
        yield from _demo_plan()
        return

    from groq import Groq

    user_msg = question
    sys_prompt = SYSTEM_PROMPT
    if literature:
        sys_prompt += (
            "\n\nYou are also given a few relevant arXiv references below. "
            "Reference them inline as [1], [2], … in your Hypothesis or Method "
            "where they directly justify a choice. Do NOT invent citations.\n\n"
            "REFERENCES:\n" + literature
        )

    client = Groq(api_key=settings.groq_api_key)
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
        temperature=0.3,
        max_tokens=900,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _demo_plan() -> Iterator[str]:
    """Hard-coded plan used when no API key is configured. Same shape as the real one."""
    text = (
        "**Hypothesis**: LoRA rank 64 will improve GSM8K accuracy on Pythia-160M "
        "by 5–15 percentage points over rank 8, since math reasoning needs more "
        "expressive adapters.\n\n"
        "**Method**:\n"
        "1. Load `EleutherAI/pythia-160m` and `gsm8k:main:train[:100]`.\n"
        "2. Fine-tune with LoRA r=8, alpha=16, lr=2e-4, 200 steps, batch size 4.\n"
        "3. Fine-tune with LoRA r=64, alpha=128, same other hyperparameters.\n"
        "4. Evaluate both on `gsm8k:main:test[:50]` (exact-match accuracy).\n"
        "5. Plot loss curves side-by-side; bar-chart final accuracies.\n\n"
        "**Metrics**: Exact-match accuracy (%), training loss curve.\n\n"
        "**Estimated time**: ~7 minutes on a Mac (no GPU)."
    )
    # simulate token streaming with variable pacing — feels more like real LLM
    import random
    import time

    for word in text.split(" "):
        yield word + " "
        # mix of fast bursts and brief pauses
        time.sleep(random.uniform(0.005, 0.04))
