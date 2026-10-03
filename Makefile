# For macOS/Linux (Windows users: use tasks.ps1)
PY := .venv/bin/python

setup:
	uv venv .venv --python 3.11
	uv pip install --python $(PY) -r requirements.txt
	uv pip install --python $(PY) -e .

lint:
	$(PY) -m ruff check .
	$(PY) -m black --check .

test:
	$(PY) -m pytest

app:
	$(PY) -m streamlit run app/main.py

api:
	$(PY) -m uvicorn api.main:app --reload

.PHONY: setup lint test app api
