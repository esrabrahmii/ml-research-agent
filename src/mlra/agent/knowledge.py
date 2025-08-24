"""Knowledge Graph — every experiment becomes a node.

Persistence:
    Each run already gets its own folder under `runs/<id>/`. We add a
    `metadata.json` there. The full graph is built by scanning all run dirs.

Edge logic (between two runs):
    weight = 3 · (#shared datasets)
           + 2 · (#shared models)
           + 1 · (#shared techniques)
    edges with weight ≥ 2 are drawn.

Rendering:
    networkx spring layout → plotly Scatter for an interactive graph.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


# ─── Persistence ──────────────────────────────────────────────────────────────

def save_node(run_dir: Path, metadata: dict) -> Path:
    """Write the node's metadata.json into its run dir."""
    out = run_dir / "metadata.json"
    out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def load_all_nodes(runs_dir: Path = RUNS_DIR) -> list[dict]:
    """Read every runs/<id>/metadata.json. Newest last (sorted by timestamp)."""
    if not runs_dir.exists():
        return []
    nodes: list[dict] = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            nodes.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    nodes.sort(key=lambda n: n.get("timestamp", ""))
    return nodes


# ─── Edge computation ─────────────────────────────────────────────────────────

def compute_edges(nodes: list[dict], min_weight: int = 2) -> list[dict]:
    """Pairwise connect nodes that share enough tags."""
    edges: list[dict] = []
    for a, b in combinations(nodes, 2):
        shared_d = set(a.get("datasets") or []) & set(b.get("datasets") or [])
        shared_m = set(a.get("models") or []) & set(b.get("models") or [])
        shared_t = set(a.get("techniques") or []) & set(b.get("techniques") or [])
        weight = 3 * len(shared_d) + 2 * len(shared_m) + len(shared_t)
        if weight >= min_weight:
            edges.append(
                {
                    "from": a["run_id"],
                    "to": b["run_id"],
                    "weight": weight,
                    "shared": {
                        "datasets": sorted(shared_d),
                        "models": sorted(shared_m),
                        "techniques": sorted(shared_t),
                    },
                }
            )
    return edges


# ─── Rendering ────────────────────────────────────────────────────────────────

DOMAIN_COLOR = {
    "medical": "#E45756",
    "botany": "#54A24B",
    "vision": "#4C78A8",
    "regression": "#F58518",
    "synthetic": "#B279A2",
    "openml": "#9D755D",
    "other": "#999999",
}

STATUS_BORDER = {
    "success": "#1f77b4",
    "recovered": "#ff7f0e",
    "failed": "#d62728",
}


def render_graph(nodes: list[dict], edges: list[dict], highlight_id: str | None = None):
    """Return a plotly Figure of the knowledge graph."""
    import networkx as nx
    import plotly.graph_objects as go

    G = nx.Graph()
    for n in nodes:
        G.add_node(n["run_id"], **n)
    for e in edges:
        G.add_edge(e["from"], e["to"], weight=e["weight"], shared=e["shared"])

    if len(G.nodes) == 0:
        # empty placeholder
        fig = go.Figure()
        fig.add_annotation(
            text="No experiments yet — run one above to start the graph.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
        )
        fig.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white")
        return fig

    # Layout — seeded so the graph is stable as nodes are added
    pos = nx.spring_layout(G, seed=42, k=1.2)

    # Edges (draw thicker for higher weights)
    edge_traces = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        shared = data.get("shared", {})
        labels = []
        for kind in ("datasets", "models", "techniques"):
            for tag in shared.get(kind, []):
                labels.append(tag)
        hover = ", ".join(labels) if labels else "shared tags"
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=0.8 + 0.6 * data.get("weight", 1), color="#bbb"),
                hoverinfo="text",
                hovertext=hover,
                showlegend=False,
            )
        )

    # Nodes
    node_x, node_y = [], []
    node_colors, node_borders, node_sizes = [], [], []
    node_text, node_hover = [], []

    for nid in G.nodes():
        x, y = pos[nid]
        node_x.append(x)
        node_y.append(y)
        d = G.nodes[nid]
        domain = d.get("primary_domain", "other")
        status = d.get("status", "success")
        node_colors.append(DOMAIN_COLOR.get(domain, "#999"))
        node_borders.append(STATUS_BORDER.get(status, "#1f77b4"))
        size = 18 + 4 * (d.get("n_plots") or 0)
        if highlight_id and nid == highlight_id:
            size += 10
        node_sizes.append(size)
        # short label: first 35 chars of question
        q = (d.get("question") or "").strip()
        node_text.append(q[:30] + ("…" if len(q) > 30 else ""))
        # rich hover
        models = ", ".join(d.get("models") or []) or "—"
        datasets = ", ".join(d.get("datasets") or []) or "—"
        techniques = ", ".join((d.get("techniques") or [])[:4])
        result = (d.get("result_line") or "").replace("RESULT:", "").strip()
        verdict = d.get("verdict", "—")
        node_hover.append(
            f"<b>{q[:60]}</b><br>"
            f"Domain: {domain}<br>"
            f"Datasets: {datasets}<br>"
            f"Models: {models}<br>"
            f"Techniques: {techniques}<br>"
            f"Result: {result[:80]}<br>"
            f"Verifier: {verdict}<br>"
            f"Status: {status} · plots: {d.get('n_plots', 0)} · retries: {d.get('n_retries', 0)}"
        )

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="bottom center",
        textfont=dict(size=10, color="#333"),
        hovertext=node_hover,
        hoverinfo="text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2.5, color=node_borders),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 1.4]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=520,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


def domain_legend_md() -> str:
    """A small markdown legend the UI can display under the graph."""
    color_dots = " · ".join(
        f"<span style='color:{c}'>●</span> {label}" for label, c in DOMAIN_COLOR.items()
    )
    border = " · ".join(
        f"<span style='border:2px solid {c};border-radius:50%;padding:0 6px'>{s}</span>"
        for s, c in STATUS_BORDER.items()
    )
    return (
        f"**Node fill (domain):** {color_dots}<br>"
        f"**Border (status):** {border}<br>"
        "**Edge thickness:** stronger when runs share more datasets/models/techniques. "
        "**Node size:** scaled by plot count. **Bigger node:** the run you just executed."
    )
