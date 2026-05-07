"""ML Research Agent — Streamlit demo UI."""
from __future__ import annotations

import sys
import time
from pathlib import Path

# make src/ importable when run via `streamlit run app/streamlit_app.py`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st  # noqa: E402

from mlra.agent.coder import stream_code  # noqa: E402
from mlra.agent.critic import diagnose_failure, stream_fix  # noqa: E402
from mlra.agent.executor import RUNS_DIR, execute, strip_code_fences  # noqa: E402
from mlra.agent.extractor import extract_metadata  # noqa: E402
from mlra.agent.knowledge import (  # noqa: E402
    compute_edges,
    domain_legend_md,
    load_all_nodes,
    render_graph,
    save_node,
)
from mlra.agent.literature import format_for_prompt  # noqa: E402
from mlra.agent.literature import search as search_arxiv
from mlra.agent.planner import stream_plan  # noqa: E402
from mlra.agent.reporter import ReportContext, stream_report  # noqa: E402
from mlra.agent.verifier import extract_result_line, verify  # noqa: E402
from mlra.config import settings  # noqa: E402

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="ML Research Agent", page_icon="🧪", layout="wide")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 ML Research Agent")
    st.caption("Autonomous ML experimentation")
    st.divider()
    st.markdown("**Models**")
    st.code(f"plan/code/critic: {settings.groq_model}", language="text")
    st.code(f"verifier:         {settings.gemini_model}", language="text")
    st.markdown("**Pipeline**")
    st.markdown(
        "1. **📚 Literature** _(arXiv)_  \n"
        "2. **🧠 Plan** _(Groq)_  \n"
        "3. **✍️ Code** _(Groq)_  \n"
        "4. **💻 Execute** _(sandbox)_  \n"
        "5. **👁️ Verify plots** _(Gemini)_  \n"
        "6. **🔁 Critic** _(if needed)_  \n"
        "7. **📄 Final Report** _(Groq)_  \n"
        "8. **🕸️ Knowledge graph**"
    )
    st.divider()
    # Show how many experiments are already in the graph
    _existing = load_all_nodes()
    st.caption(f"Knowledge graph: **{len(_existing)}** experiments stored")
    st.divider()
    if not settings.groq_api_key:
        st.error("No GROQ_API_KEY")
    else:
        st.success("Connected to Groq")
    if settings.google_api_key:
        st.success("Connected to Gemini")
    else:
        st.warning("Verifier disabled (no GOOGLE_API_KEY)")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🧪 ML Research Agent")
st.caption("Type any ML research question. The agent reads literature, plans, runs, verifies, self-corrects, and reports.")

DEFAULT_Q = "How does training set size affect logistic regression accuracy on the breast cancer dataset?"

question = st.text_area(
    "Research question",
    value=DEFAULT_Q,
    height=80,
    label_visibility="collapsed",
    placeholder="What ML question would you like answered?",
)

col_run, col_arxiv = st.columns([1, 2])
with col_run:
    run = st.button("▶  Run Experiment", type="primary", disabled=not question.strip())
with col_arxiv:
    use_arxiv = st.checkbox("📚 Cite arXiv literature", value=True)


# ─── Pipeline helpers ─────────────────────────────────────────────────────────

def run_executor(code: str) -> tuple[list[str], list[Path], int, str, float]:
    """Run the executor; render stdout + plots live; return everything we need."""
    log_lines: list[str] = []
    shown_plots: list[Path] = []
    return_code = -1
    run_id = ""

    exec_status = st.status("💻 Running in sandbox…", expanded=True, state="running")
    with exec_status:
        col_log, col_plot = st.columns([1, 1])
        with col_log:
            st.markdown("**stdout**")
            log_holder = st.empty()
        with col_plot:
            st.markdown("**plots**")
            plot_placeholder = st.empty()

        t0 = time.time()
        for kind, payload in execute(code):
            if kind == "run_id":
                run_id = str(payload)
            elif kind == "stdout":
                log_lines.append(str(payload))
                log_holder.code("\n".join(log_lines[-30:]), language="text")
            elif kind == "stderr":
                log_lines.append(f"⚠ {payload}")
                log_holder.code("\n".join(log_lines[-30:]), language="text")
            elif kind == "plot":
                p = payload  # type: ignore[assignment]
                if p not in shown_plots:
                    shown_plots.append(p)  # type: ignore[arg-type]
                    with plot_placeholder.container():
                        for img in shown_plots:
                            st.image(str(img), use_container_width=True)
                            st.caption(img.name)
            elif kind == "tick":
                exec_status.update(
                    label=f"💻 Running in sandbox… ({float(payload):.0f}s)",  # type: ignore[arg-type]
                    state="running",
                )
            elif kind == "done":
                return_code = payload.return_code  # type: ignore[union-attr]

        elapsed = time.time() - t0
        if return_code == 0:
            exec_status.update(label=f"✓ Run complete ({elapsed:.1f}s · `{run_id}`)", state="complete")
        else:
            exec_status.update(
                label=f"✗ Run failed (exit {return_code}, {elapsed:.1f}s · `{run_id}`)", state="error"
            )

    return log_lines, shown_plots, return_code, run_id, elapsed


def run_verifier(plots: list[Path], log_lines: list[str]) -> tuple[list[str], list[str]]:
    """Run Gemini Vision verifier on each plot. Returns (verdicts, visuals)."""
    if not settings.google_api_key or not plots:
        return [], []
    verdicts: list[str] = []
    visuals: list[str] = []
    verify_status = st.status("👁️ Verifying plots with Gemini Vision…", expanded=True, state="running")
    with verify_status:
        claim = extract_result_line(log_lines)
        for plot in plots:
            v = verify(plot, claim)
            verdicts.append(v.verdict)
            visuals.append(v.visual)
            emoji = {"SUPPORTED": "✅", "UNCLEAR": "❓", "CONTRADICTED": "❌", "ERROR": "⚠️"}.get(v.verdict, "•")
            with st.container(border=True):
                st.markdown(f"### {emoji} `{plot.name}` — **{v.verdict}**")
                if v.visual:
                    st.markdown(f"**Visual evidence:** {v.visual}")
                if v.note:
                    st.markdown(f"**Verifier's note:** {v.note}")
        bad = sum(1 for v in verdicts if v == "CONTRADICTED")
        if bad:
            verify_status.update(label=f"❌ {bad} plot(s) contradict the claim", state="error")
        else:
            verify_status.update(label=f"✓ Verified {len(plots)} plot(s)", state="complete")
    return verdicts, visuals


# ─── Pipeline ─────────────────────────────────────────────────────────────────
if run:
    st.markdown("")

    # 0. Literature
    literature_text: str | None = None
    papers: list = []
    if use_arxiv:
        with st.status("📚 Searching arXiv…", expanded=True, state="running") as lit_status:
            t0 = time.time()
            papers = search_arxiv(question, max_results=4)
            literature_text = format_for_prompt(papers)
            if papers:
                for p in papers:
                    st.markdown(f"- **[{p.short_cite()}]** *{p.title}* — [arXiv:{p.arxiv_id}]({p.url})")
                lit_status.update(label=f"✓ Found {len(papers)} papers ({time.time()-t0:.1f}s)", state="complete")
            else:
                lit_status.update(label="⚠ No papers found", state="complete")

    col_left, col_right = st.columns(2)

    # 1. Plan
    with col_left:
        with st.status("🧠 Planning…", expanded=True, state="running") as plan_status:
            plan_holder = st.empty()
            plan_text = ""
            t0 = time.time()
            for chunk in stream_plan(question, literature=literature_text):
                plan_text += chunk
                plan_holder.markdown(plan_text + "▌")
            plan_holder.markdown(plan_text)
            plan_elapsed = time.time() - t0
            plan_status.update(label=f"✓ Plan ready ({plan_elapsed:.1f}s)", state="complete")

    # 2. Code
    with col_right:
        with st.status("✍️ Generating code…", expanded=True, state="running") as code_status:
            code_holder = st.empty()
            code_text = ""
            t0 = time.time()
            for chunk in stream_code(question, plan_text):
                code_text += chunk
                code_holder.code(code_text + "▌", language="python")
            code_holder.code(strip_code_fences(code_text), language="python")
            code_elapsed = time.time() - t0
            code_status.update(label=f"✓ Code ready ({code_elapsed:.1f}s)", state="complete")

    # 3. Execute
    code = strip_code_fences(code_text)
    log_lines, plots, return_code, run_id, exec_elapsed = run_executor(code)

    # 4. Verify
    verdicts, visuals = run_verifier(plots, log_lines)

    # 5. Critic + retry — fires on exec failure OR all-contradicted verdicts
    n_retries = 0
    failure_msg = diagnose_failure(return_code, log_lines, verdicts)
    if failure_msg:
        with st.status("🔁 Critic — diagnosing failure and proposing fix…", expanded=True, state="running") as crit_status:
            st.markdown("**Detected issue:**")
            st.warning(failure_msg[:600] + ("…" if len(failure_msg) > 600 else ""))
            st.markdown("**Fixed code (streaming):**")
            fix_holder = st.empty()
            fixed_code = ""
            t0 = time.time()
            for chunk in stream_fix(question, plan_text, code, failure_msg):
                fixed_code += chunk
                fix_holder.code(fixed_code + "▌", language="python")
            fixed_code = strip_code_fences(fixed_code)
            fix_holder.code(fixed_code, language="python")
            crit_elapsed = time.time() - t0
            crit_status.update(label=f"✓ Fix proposed ({crit_elapsed:.1f}s) — re-running…", state="complete")

        # Re-execute
        log_lines, plots, return_code, run_id, exec_elapsed = run_executor(fixed_code)
        # Re-verify the new plots
        verdicts, visuals = run_verifier(plots, log_lines)
        n_retries = 1
        code = fixed_code

    # 6. Final report
    st.markdown("")
    with st.status("📄 Writing final report…", expanded=True, state="running") as rep_status:
        ctx = ReportContext(
            question=question,
            plan=plan_text,
            stdout="\n".join(log_lines),
            verifier_verdicts=verdicts,
            verifier_visuals=visuals,
            references=[f"{p.short_cite()} — {p.url}" for p in papers],
            n_plots=len(plots),
            n_retries=n_retries,
        )
        report_holder = st.empty()
        report_text = ""
        t0 = time.time()
        for chunk in stream_report(ctx):
            report_text += chunk
            report_holder.markdown(report_text + "▌")
        report_holder.markdown(report_text)
        rep_elapsed = time.time() - t0
        rep_status.update(label=f"✓ Report ready ({rep_elapsed:.1f}s)", state="complete")

    # 7. Knowledge graph — extract metadata, save it, render the growing graph
    metadata = extract_metadata(
        run_id=run_id,
        question=question,
        code=code,
        stdout="\n".join(log_lines),
        verdicts=verdicts,
        papers=[p.short_cite() for p in papers],
        n_plots=len(plots),
        n_retries=n_retries,
        return_code=return_code,
    )
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        save_node(run_dir, metadata)
    all_nodes = load_all_nodes()
    edges = compute_edges(all_nodes)
    st.markdown("")
    st.markdown("## 🕸️ Knowledge Graph")
    g_col_a, g_col_b = st.columns([3, 2])
    with g_col_a:
        fig = render_graph(all_nodes, edges, highlight_id=run_id)
        st.plotly_chart(fig, use_container_width=True)
    with g_col_b:
        st.markdown("**This run's tags**")
        st.markdown(f"- Domain: `{metadata['primary_domain']}`")
        st.markdown("- Datasets: " + (", ".join(f"`{d}`" for d in metadata["datasets"]) or "—"))
        st.markdown("- Models: " + (", ".join(f"`{m}`" for m in metadata["models"]) or "—"))
        st.markdown("- Techniques: " + (", ".join(f"`{t}`" for t in metadata["techniques"][:6]) or "—"))
        st.markdown(f"- Status: `{metadata['status']}` · Verifier: `{metadata['verdict']}`")
        st.markdown("---")
        st.markdown(f"**Total experiments:** {len(all_nodes)}  \n**Connections:** {len(edges)}")
    st.markdown(domain_legend_md(), unsafe_allow_html=True)

    # Metrics
    metrics: list[tuple[str, str | int]] = []
    if use_arxiv:
        metrics.append(("Papers", len(papers)))
    metrics += [
        ("Plan", f"{plan_elapsed:.1f}s"),
        ("Code", f"{code_elapsed:.1f}s"),
        ("Exec", f"{exec_elapsed:.1f}s"),
        ("Plots", len(plots)),
        ("Retries", n_retries),
        ("Graph nodes", len(all_nodes)),
    ]
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)

    with st.expander("Coming next in the pipeline"):
        st.markdown(
            "- 🕸️ **Knowledge graph** — every experiment becomes a node the agent can reference later\n"
            "- 🦆 **DuckDB analytics** — SQL queries over your run history\n"
            "- 🎯 **Recommender** — \"based on past runs, try X next\"\n"
            "- 🎙️ **Voice input** + ☁️ cloud deploy"
        )
