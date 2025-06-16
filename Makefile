.PHONY: install run test lint clean

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# ─── Setup ──────────────────────────────────────────────────────────────────
install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"
	@echo ""
	@echo "✓ Installed. Now copy .env.example to .env and add your Groq key."
	@echo "  Get one free at https://console.groq.com/keys"

# ─── Run ────────────────────────────────────────────────────────────────────
run:
	$(VENV)/bin/streamlit run app/streamlit_app.py

# ─── Quality ────────────────────────────────────────────────────────────────
test:
	$(VENV)/bin/pytest tests/unit/ -v

lint:
	$(VENV)/bin/ruff check src/ tests/ app/

format:
	$(VENV)/bin/ruff format src/ tests/ app/

# ─── Housekeeping ───────────────────────────────────────────────────────────
clean:
	rm -rf $(VENV) .pytest_cache .mypy_cache .ruff_cache *.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
