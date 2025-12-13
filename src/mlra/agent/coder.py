"""Code generator — turns a research plan into a runnable Python script.

Constraints baked into the prompt:
  - Runs in <60s on a Mac CPU
  - Uses scikit-learn / numpy / matplotlib only
  - No plt.show(); writes plots to disk via plt.savefig()
  - Prints key results to stdout
"""
from __future__ import annotations

from collections.abc import Iterator

from mlra.config import settings

CODE_SYSTEM_PROMPT = """You are an expert ML research engineer. Given a research question and a plan, write a single, complete, runnable Python script that executes the experiment.

STRICT RULES:
1. Use ONLY: numpy, pandas, scikit-learn, matplotlib (no torch, transformers, datasets — those are too heavy for this sandbox).
2. Total runtime MUST be under 60 seconds on a CPU.
3. If the plan needs LLMs/transformers/HuggingFace: pivot to a SCIKIT-LEARN ANALOGY that mirrors the same scientific question (e.g. compare low-rank matrix approximations instead of LoRA ranks; compare logistic regression hyperparams instead of LLM hyperparams). Note the analogy in stdout.
4. Save every plot via `plt.savefig("plot_N.png", dpi=120, bbox_inches='tight')` where N is 1, 2, 3... NEVER use plt.show().
5. Print structured results clearly. End with a one-line `RESULT:` summary that names the headline number.
6. Wrap the whole experiment in try/except — on error print a clear `ERROR: <msg>` line.

CORRECTNESS RULES (avoid common bugs):
A. ALWAYS scale features with `StandardScaler` before LogisticRegression / SVM / kNN / linear models — prevents convergence failures.
B. ALWAYS pass `max_iter=5000` to LogisticRegression — avoids ConvergenceWarning.
C. When using `train_test_split`, use a FIXED `test_size=0.2` and `random_state=42`. NEVER vary test_size dynamically inside a loop.
D. When sweeping a hyperparameter (e.g. training set size), do a SINGLE train/test split once, then index into the training set: `X_train_subset = X_train[:n]`. Never re-split per iteration.
E. Choose dataset sizes that make every iteration meaningful — don't sweep n=10 (too noisy).
F. Print at most ~15 lines of stdout. Headlines, not verbose logs.

Output ONLY the Python code. No markdown fences, no commentary. The code should run as-is.
"""


def stream_code(question: str, plan: str) -> Iterator[str]:
    """Stream the generated Python script token-by-token."""
    if not settings.groq_api_key:
        yield from _demo_code()
        return

    from groq import Groq

    client = Groq(api_key=settings.groq_api_key)
    prompt = f"Research question:\n{question}\n\nPlan:\n{plan}\n\nWrite the script now."
    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        stream=True,
        temperature=0.2,
        max_tokens=2000,
    )
    for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def _demo_code() -> Iterator[str]:
    """Hard-coded code for demo mode."""
    import random
    import time

    code = '''import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

try:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    sizes = [50, 100, 200, 350]
    accs = []
    for n in sizes:
        clf = LogisticRegression(max_iter=2000)
        clf.fit(X_train[:n], y_train[:n])
        acc = clf.score(X_test, y_test)
        accs.append(acc)
        print(f"n={n}: accuracy={acc:.3f}")

    plt.figure(figsize=(6, 4))
    plt.plot(sizes, accs, "o-")
    plt.xlabel("Training set size"); plt.ylabel("Test accuracy")
    plt.title("How does training size affect logistic regression?")
    plt.grid(alpha=0.3)
    plt.savefig("plot_1.png", dpi=120, bbox_inches="tight")
    print(f"\\nRESULT: accuracy goes from {accs[0]:.3f} (n=50) to {accs[-1]:.3f} (n=350) — gain of {accs[-1]-accs[0]:.3f}")
except Exception as e:
    print(f"ERROR: {e}")
'''
    for token in code.split(" "):
        yield token + " "
        time.sleep(random.uniform(0.003, 0.02))
