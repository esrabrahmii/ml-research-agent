"""Local subprocess executor.

Each run gets its own directory under `runs/`. The agent's generated code is
saved there and executed with cwd set to that directory, so any plots saved
via `plt.savefig("plot_1.png")` land in the run's folder.

Uses a reader thread + queue so the UI never blocks waiting on subprocess output.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


class RunResult(NamedTuple):
    run_id: str
    run_dir: Path
    code_path: Path
    plots: list[Path]
    return_code: int


def new_run_dir() -> tuple[str, Path]:
    run_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap code in ```python ... ``` despite instructions. Strip if present."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3].rstrip()
    return text


def execute(code: str, timeout: int = 90) -> Iterator[tuple[str, object]]:
    """Run the code in a subprocess. Yields events:
      ("run_id", str), ("stdout", str), ("stderr", str),
      ("plot", Path), ("tick", float elapsed), ("done", RunResult)
    """
    code = strip_code_fences(code)
    run_id, run_dir = new_run_dir()
    code_path = run_dir / "experiment.py"
    code_path.write_text(code, encoding="utf-8")

    yield "run_id", run_id
    yield "stdout", f"[starting subprocess in runs/{run_id}/]"

    env = {
        **os.environ,
        "MPLBACKEND": "Agg",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        # Silence sklearn / numpy / matplotlib warnings — keep stdout clean
        "PYTHONWARNINGS": "ignore",
    }
    proc = subprocess.Popen(
        [sys.executable, "-u", str(code_path)],
        cwd=str(run_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    # Reader thread drains stdout/stderr into a queue so the main loop
    # can interleave with plot-file polling and heartbeats.
    q: queue.Queue = queue.Queue()

    def _reader() -> None:
        assert proc.stdout is not None
        for line in iter(proc.stdout.readline, ""):
            q.put(("stdout", line.rstrip("\n")))
        q.put(("eof", None))

    threading.Thread(target=_reader, daemon=True).start()

    seen_plots: set[Path] = set()
    start = time.time()
    last_tick = start
    eof_seen = False

    while True:
        elapsed = time.time() - start

        # 1) drain whatever's in the queue
        try:
            while True:
                kind, payload = q.get_nowait()
                if kind == "eof":
                    eof_seen = True
                    break
                yield kind, payload
        except queue.Empty:
            pass

        # 2) poll for new plots
        for p in sorted(run_dir.glob("plot_*.png")):
            if p not in seen_plots:
                seen_plots.add(p)
                yield "plot", p

        # 3) heartbeat tick once a second so the UI shows progress
        if time.time() - last_tick >= 1.0:
            last_tick = time.time()
            yield "tick", elapsed

        # 4) exit conditions
        if eof_seen:
            break
        if elapsed >= timeout:
            proc.kill()
            yield "stderr", f"[killed after {timeout}s timeout]"
            break

        time.sleep(0.1)

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()

    # final sweep of plots in case the script wrote them just before exit
    for p in sorted(run_dir.glob("plot_*.png")):
        if p not in seen_plots:
            seen_plots.add(p)
            yield "plot", p

    yield "done", RunResult(
        run_id=run_id,
        run_dir=run_dir,
        code_path=code_path,
        plots=sorted(seen_plots),
        return_code=proc.returncode if proc.returncode is not None else -1,
    )
