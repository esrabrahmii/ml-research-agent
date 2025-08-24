"""Tests for the knowledge graph — persistence + edge logic."""
import json

from mlra.agent.knowledge import compute_edges, load_all_nodes, render_graph, save_node


def make_node(rid, datasets=(), models=(), techniques=(), domain="medical"):
    return {
        "run_id": rid,
        "question": f"q{rid}",
        "datasets": list(datasets),
        "models": list(models),
        "techniques": list(techniques),
        "primary_domain": domain,
        "result_line": f"RESULT: {rid}",
        "verdict": "SUPPORTED",
        "n_plots": 1,
        "n_retries": 0,
        "status": "success",
        "n_papers": 0,
        "timestamp": f"2026-05-07T10:00:0{rid[-1]}Z",
    }


def test_compute_edges_strong_connection():
    a = make_node("a", datasets=["breast_cancer"], models=["LogisticRegression"])
    b = make_node("b", datasets=["breast_cancer"], models=["LogisticRegression"])
    edges = compute_edges([a, b])
    assert len(edges) == 1
    assert edges[0]["weight"] >= 5  # 3 (dataset) + 2 (model)


def test_compute_edges_weak_no_connection():
    a = make_node("a", techniques=["StandardScaler"])
    b = make_node("b", techniques=["StandardScaler"])
    # Only 1 shared technique → weight 1, below the default min_weight=2
    assert compute_edges([a, b]) == []


def test_compute_edges_two_techniques_connect():
    a = make_node("a", techniques=["StandardScaler", "PCA"])
    b = make_node("b", techniques=["StandardScaler", "PCA"])
    edges = compute_edges([a, b])
    assert len(edges) == 1
    assert edges[0]["weight"] == 2


def test_save_and_load(tmp_path):
    run_dir = tmp_path / "abc123"
    run_dir.mkdir()
    md = make_node("abc123")
    save_node(run_dir, md)
    assert (run_dir / "metadata.json").exists()
    loaded = json.loads((run_dir / "metadata.json").read_text())
    assert loaded["run_id"] == "abc123"


def test_load_all_nodes(tmp_path):
    for rid in ("a1", "b2"):
        d = tmp_path / rid
        d.mkdir()
        save_node(d, make_node(rid))
    nodes = load_all_nodes(tmp_path)
    assert len(nodes) == 2


def test_render_empty_graph_doesnt_crash():
    fig = render_graph([], [])
    assert fig is not None


def test_render_with_nodes_doesnt_crash():
    a = make_node("a", datasets=["breast_cancer"], models=["LogisticRegression"])
    b = make_node("b", datasets=["breast_cancer"], models=["SVC"])
    edges = compute_edges([a, b])
    fig = render_graph([a, b], edges, highlight_id="b")
    assert fig is not None
