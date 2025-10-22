"""Tests for the metadata extractor — deterministic, no API calls."""
from mlra.agent.extractor import extract_metadata, extract_result_line

SAMPLE_CODE = """
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train_scaled, y_train)
"""


def test_extract_metadata_basic():
    md = extract_metadata(
        run_id="abc123",
        question="Does scaling help?",
        code=SAMPLE_CODE,
        stdout="loading\nRESULT: scaled accuracy 0.97 vs unscaled 0.92",
        verdicts=["SUPPORTED"],
        papers=["Smith 2024"],
        n_plots=1,
        n_retries=0,
        return_code=0,
    )
    assert md["run_id"] == "abc123"
    assert "breast_cancer" in md["datasets"]
    assert "LogisticRegression" in md["models"]
    assert "StandardScaler" in md["techniques"]
    assert "train_test_split" in md["techniques"]
    assert md["primary_domain"] == "medical"
    assert "RESULT" in md["result_line"]
    assert md["status"] == "success"
    assert md["verdict"] == "SUPPORTED"


def test_extract_failure_status():
    md = extract_metadata(
        run_id="abc",
        question="q",
        code="",
        stdout="ERROR: oops",
        verdicts=[],
        papers=[],
        n_plots=0,
        n_retries=0,
        return_code=1,
    )
    assert md["status"] == "failed"


def test_extract_recovered_status():
    md = extract_metadata(
        run_id="abc",
        question="q",
        code=SAMPLE_CODE,
        stdout="RESULT: x",
        verdicts=["SUPPORTED"],
        papers=[],
        n_plots=1,
        n_retries=1,
        return_code=0,
    )
    assert md["status"] == "recovered"


def test_extract_result_line_helper():
    assert extract_result_line("a\nb\nRESULT: thing") == "RESULT: thing"
    assert extract_result_line("no result here") == ""
