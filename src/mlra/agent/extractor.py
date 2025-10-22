"""Extract structured metadata from a finished experiment.

The metadata becomes a node in the knowledge graph: dataset(s) used, model
class(es), preprocessing techniques, headline result. Pure regex/heuristics —
no LLM call, so it's fast and deterministic.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

# Known sklearn things we want to track. Order matters for display.
DATASET_PATTERNS = [
    ("iris", r"load_iris\b"),
    ("breast_cancer", r"load_breast_cancer\b"),
    ("wine", r"load_wine\b"),
    ("digits", r"load_digits\b"),
    ("diabetes", r"load_diabetes\b"),
    ("california_housing", r"fetch_california_housing\b"),
    ("synthetic_classification", r"make_classification\b"),
    ("synthetic_regression", r"make_regression\b"),
    ("synthetic_blobs", r"make_blobs\b"),
    ("openml", r"fetch_openml\b"),
]

MODEL_PATTERNS = [
    ("LogisticRegression", r"\bLogisticRegression\b"),
    ("LinearRegression", r"\bLinearRegression\b"),
    ("Ridge", r"\bRidge\b\("),
    ("Lasso", r"\bLasso\b\("),
    ("ElasticNet", r"\bElasticNet\b\("),
    ("RandomForestClassifier", r"\bRandomForestClassifier\b"),
    ("RandomForestRegressor", r"\bRandomForestRegressor\b"),
    ("GradientBoostingClassifier", r"\bGradientBoostingClassifier\b"),
    ("GradientBoostingRegressor", r"\bGradientBoostingRegressor\b"),
    ("SVC", r"\bSVC\b\("),
    ("SVR", r"\bSVR\b\("),
    ("KNeighborsClassifier", r"\bKNeighborsClassifier\b"),
    ("KNeighborsRegressor", r"\bKNeighborsRegressor\b"),
    ("DecisionTreeClassifier", r"\bDecisionTreeClassifier\b"),
    ("MLPClassifier", r"\bMLPClassifier\b"),
    ("KMeans", r"\bKMeans\b"),
]

TECHNIQUE_PATTERNS = [
    ("StandardScaler", r"\bStandardScaler\b"),
    ("MinMaxScaler", r"\bMinMaxScaler\b"),
    ("PCA", r"\bPCA\b"),
    ("train_test_split", r"\btrain_test_split\b"),
    ("cross_val_score", r"\bcross_val_score\b|\bKFold\b|\bStratifiedKFold\b"),
    ("GridSearchCV", r"\bGridSearchCV\b"),
    ("Pipeline", r"\bPipeline\b\("),
    ("OneHotEncoder", r"\bOneHotEncoder\b"),
    ("L1_regularization", r"penalty\s*=\s*['\"]l1['\"]"),
    ("L2_regularization", r"penalty\s*=\s*['\"]l2['\"]"),
]

# topic colour for graph display: maps a node to a "domain"
DOMAIN_BY_DATASET = {
    "breast_cancer": "medical",
    "diabetes": "medical",
    "iris": "botany",
    "wine": "botany",
    "digits": "vision",
    "california_housing": "regression",
    "synthetic_classification": "synthetic",
    "synthetic_regression": "synthetic",
    "synthetic_blobs": "synthetic",
    "openml": "openml",
}


def _find_all(patterns: list[tuple[str, str]], text: str) -> list[str]:
    found = []
    for label, pat in patterns:
        if re.search(pat, text):
            found.append(label)
    return found


def extract_result_line(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.strip().upper().startswith("RESULT:"):
            return line.strip()
    return ""


def extract_metadata(
    *,
    run_id: str,
    question: str,
    code: str,
    stdout: str,
    verdicts: list[str],
    papers: list[str],
    n_plots: int,
    n_retries: int,
    return_code: int,
) -> dict:
    """Return a serializable metadata dict ready to drop into knowledge_graph.json."""
    datasets = _find_all(DATASET_PATTERNS, code)
    models = _find_all(MODEL_PATTERNS, code)
    techniques = _find_all(TECHNIQUE_PATTERNS, code)

    domains = sorted({DOMAIN_BY_DATASET.get(d, "other") for d in datasets}) or ["other"]
    primary_domain = domains[0]

    if return_code != 0:
        status = "failed"
    elif n_retries > 0:
        status = "recovered"
    else:
        status = "success"

    primary_verdict = verdicts[0] if verdicts else "—"

    return {
        "run_id": run_id,
        "question": question,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "models": models,
        "techniques": techniques,
        "primary_domain": primary_domain,
        "result_line": extract_result_line(stdout),
        "verdict": primary_verdict,
        "n_plots": n_plots,
        "n_retries": n_retries,
        "status": status,
        "n_papers": len(papers),
    }
