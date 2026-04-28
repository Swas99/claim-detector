# L2Labs Claim Detection — Master Plan

## Context

L2Labs take-home: fine-tuned LM for sentence-level claim detection. API receives a sentence, returns `{is_claim: bool, confidence: float}`. Multi-session build. Design and explanation matter more than raw accuracy.

**Location:** `/Users/swas/Desktop/L2Labs/claim-detector`
**Hardware:** Apple Silicon (MPS)
**Dataset:** ~13k sentences from github.com/VeritaResearch/claim-extraction

---

## High-Level Phases (8 phases, ~20 hours total)

### Phase 1: Foundation (~1 hour)
- Init git repo, project structure, dependencies
- Download + validate dataset
- Write data loading/preprocessing module
- **Deliverable:** `python -c "from src.data import load_dataset; print(load_dataset())"` works

### Phase 2: Model Training + Comparison (~3 hours)
Train 4 models, evaluate all on same test set:
1. **TF-IDF + XGBoost** — classical baseline (~5 sec training)
2. **DistilBERT** — lightweight, fast inference (~15 min training)
3. **BERT-base** — paper reproduction (~30 min training)
4. **DeBERTa-v3-base** — SOTA comparison (~25 min training)
- **Deliverable:** comparison table + saved model weights, metrics JSON per model

### Phase 3: Inference Engine (~3 hours)
- ONNX export for winning model
- Cascading inference pipeline (fast filter → model)
- Confidence calibration (Platt/temperature scaling)
- Semantic cache (CLS embedding → LSH → LRU)
- **Deliverable:** `ClaimDetector` class with `.predict(text)` that runs full pipeline

### Phase 4: API Server (~2 hours)
- FastAPI with POST /predict, POST /predict/batch, GET /health, GET /model/info
- POST /feedback (uncertain prediction logging)
- Pydantic schemas, input validation, CORS, rate limiting
- Structured JSON logging
- **Deliverable:** `uvicorn src.api.main:app` serves all endpoints

### Phase 5: Explainability (~2 hours)
- Token attribution (attention weights or integrated gradients)
- Claim type enrichment (optional multi-task head or rule-based)
- Confidence calibration visualization
- **Deliverable:** /predict returns attribution array alongside prediction

### Phase 6: Testing (~3 hours)
- Unit tests (model predictions on known examples)
- Integration tests (full API round-trip)
- Security tests (input validation, rate limiting)
- Load tests (Locust — throughput, latency percentiles)
- **Deliverable:** `pytest` green, Locust report in docs/

### Phase 7: Containerization (~2 hours)
- Multi-stage Dockerfile (builder → slim runtime)
- docker-compose.yml
- Health check built into container
- **Deliverable:** `docker compose up` → curl endpoints works

### Phase 8: Documentation + Notebook (~3 hours)
- Training report notebook (loss curves, confusion matrix, ROC, model comparison, OOD analysis)
- Architecture doc (production diagram + local-vs-prod mapping)
- Model card
- README (setup, usage, API docs, results summary)
- **Deliverable:** polished repo ready for review

---

## Architecture Component Mapping

```
Architecture Component          → Implementation                → Phase
─────────────────────────────────────────────────────────────────────────
Edge/Gateway (nginx, TLS)       → docs only (prod discussion)   → 8
Rate Limiting                   → slowapi in FastAPI             → 4
Auth (API key)                  → middleware + config             → 4
CORS                            → FastAPI CORSMiddleware         → 4
FastAPI Server                  → src/api/                       → 4
Pydantic Validation             → src/api/schemas.py             → 4
Semantic Cache                  → src/engine/cache.py            → 3
Fast Filter (rule-based)        → src/engine/fast_filter.py      → 3
BERT/ONNX Inference             → src/engine/model_runner.py     → 3
Confidence Calibration          → src/engine/calibration.py      → 5
Token Attribution               → src/engine/attribution.py      → 5
Claim Type Enrichment           → src/engine/claim_type.py       → 5
Prometheus Metrics              → docs only (prod discussion)    → 8
Structured Logging              → src/api/main.py                → 4
Feedback Store                  → src/feedback/store.py (SQLite) → 4
Model Training                  → src/training/                  → 2
ONNX Export                     → src/training/export_onnx.py    → 3
Docker                          → Dockerfile, compose            → 7
```

---

## Project File Structure

```
claim-detector/
├── README.md
├── HIGH_LEVEL_PLAN.md           # This plan (lives in repo for session continuity)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
├── .gitignore
│
├── docs/
│   ├── architecture.md          # Prod architecture + local-vs-prod mapping
│   ├── model_card.md
│   ├── load_test_report.md
│   └── phases/                  # Low-level plans per phase
│       ├── phase1_foundation.md
│       ├── phase2_models.md
│       ├── phase3_engine.md
│       ├── phase4_api.md
│       ├── phase5_explainability.md
│       ├── phase6_testing.md
│       ├── phase7_docker.md
│       └── phase8_docs.md
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_tfidf.py
│   │   ├── train_transformer.py
│   │   └── export_onnx.py
│   ├── engine/                  # Inference pipeline
│   │   ├── __init__.py
│   │   ├── detector.py          # ClaimDetector (orchestrator)
│   │   ├── fast_filter.py
│   │   ├── model_runner.py
│   │   ├── cache.py
│   │   ├── calibration.py
│   │   ├── attribution.py
│   │   └── claim_type.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── routes.py
│   │   └── schemas.py
│   └── feedback/
│       ├── __init__.py
│       └── store.py
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_engine.py
│   ├── test_api.py
│   ├── test_security.py
│   └── locustfile.py
│
├── notebooks/
│   └── training_report.ipynb
│
├── data/                        # .gitignored — raw dataset
│   └── README.md                # Where to get the data
│
└── models/                      # .gitignored — trained weights
    └── README.md                # Model artifacts description
```

---

## Session Strategy

Each session:
1. Read HIGH_LEVEL_PLAN.md → check current phase
2. Read the relevant docs/phases/phaseN.md for detailed plan
3. Implement
4. Commit with clear messages
5. Update phase doc with completion status

Phases are independent enough to stop/resume at any boundary.

---

## What We Build Locally vs. Discuss for Prod

| Component | Local | Prod (discussion in docs) |
|---|---|---|
| Rate limiting | slowapi (in-process) | API Gateway (Kong/nginx) |
| Auth | API key in env var | OAuth2 / JWT with rotation |
| Cache | In-memory LRU | Redis cluster |
| Logging | JSON to stdout | ELK / Loki + Grafana |
| Metrics | Log-based | Prometheus + Grafana |
| Feedback store | SQLite | PostgreSQL |
| Scaling | Single process | K8s HPA, multiple replicas |
| Model storage | Local filesystem | S3/GCS with versioning |
| TLS | None (localhost) | Cert-manager / ACM |
| CI/CD | None | GitHub Actions → GHCR → K8s |
