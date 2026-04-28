.PHONY: setup train train-tfidf train-distilbert train-bert train-modernbert fit-calibration export-onnx serve test load-test docker-build docker-up clean

PYTHON := python
VENV := .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python

# ─── Setup ────────────────────────────────────────────────────────────────────

setup:
	python -m venv $(VENV)
	$(PIP) install -e ".[dev]"
	@echo "Run: source $(VENV)/bin/activate"

download-data:
	$(PY) -m src.data.dataset --download

# ─── Training ─────────────────────────────────────────────────────────────────

train-tfidf:
	$(PY) -m src.training.train_tfidf

train-distilbert:
	$(PY) -m src.training.train_transformer --model distilbert-base-uncased --epochs 5 --batch-size 16

train-bert:
	$(PY) -m src.training.train_transformer --model bert-base-uncased --epochs 5 --batch-size 16

train-modernbert:
	$(PY) -m src.training.train_transformer --model answerdotai/ModernBERT-base --epochs 5 --batch-size 16

train: train-tfidf train-distilbert train-bert train-modernbert fit-calibration

fit-calibration:
	$(PY) -m src.training.fit_calibration

export-onnx:
	$(PY) -m src.training.export_onnx

# ─── Serve ────────────────────────────────────────────────────────────────────

serve:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	$(PY) -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ─── Test ─────────────────────────────────────────────────────────────────────

test:
	$(PY) -m pytest tests/ -v

test-cov:
	$(PY) -m pytest tests/ -v --cov=src --cov-report=html

load-test:
	$(PY) -m locust -f tests/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 60s

# ─── Docker ───────────────────────────────────────────────────────────────────

docker-build:
	docker build -t claim-detector .

docker-up:
	docker compose up -d

docker-down:
	docker compose down

# ─── Clean ────────────────────────────────────────────────────────────────────

clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
