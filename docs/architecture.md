# System Architecture

## Overview

The claim detector is a multi-model classification system that identifies whether a natural language sentence contains a verifiable factual claim. It supports three transformer models (DistilBERT, BERT, ModernBERT) plus an ensemble mode, with confidence calibration and token-level attribution.

## Local Architecture

```
    Client (curl / browser / demo UI)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                  FastAPI Server                     │
│  - Demo UI at /  (sentence + paragraph modes)      │
│  - CORS middleware                                  │
│  - Rate limiting (slowapi, 60 req/min per IP)      │
│  - Structured JSON logging (structlog)             │
│  - Pydantic input validation                       │
│                                                     │
│  Endpoints:                                         │
│    POST /predict        single sentence             │
│    POST /predict/batch  up to 100 sentences         │
│    POST /compare        all models side-by-side     │
│    POST /analyze        paragraph claim highlighter │
│    GET  /health         liveness check              │
│    GET  /model/info     metadata + eval metrics     │
│    POST /feedback       correction logging          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│              Inference Pipeline                     │
│                                                     │
│  1. CACHE CHECK ──► exact (text, model) hit?        │
│     │ miss             return cached prediction     │
│     ▼                                               │
│  2. MODEL INFERENCE                                │
│     ├── DistilBERT (default, 66M params)           │
│     ├── BERT-base (110M params)                    │
│     ├── ModernBERT (150M params)                   │
│     └── Ensemble (average of all three)            │
│     │   Models loaded lazily on first use           │
│     ▼                                               │
│  3. CONFIDENCE CALIBRATION (T=2.15)                │
│     │                                               │
│     ▼                                               │
│  4. TOKEN ATTRIBUTION (attention weights)          │
│     │                                               │
│     ▼                                               │
│  5. CACHE STORE + RETURN                           │
└───────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│              Persistence                            │
│  - Feedback store: SQLite (corrections)            │
│  - Prediction cache: in-memory dict per (text,model)│
│  - Model weights: local filesystem                 │
└───────────────────────────────────────────────────┘
```

## Production Architecture

```
                          PRODUCTION
    ┌──────────────────────────────────────────────────────────┐
    │                    EDGE / GATEWAY                         │
    │  Nginx / Traefik       TLS termination                   │
    │  Rate limiting (global) OAuth2 / JWT auth                │
    └────────────────────────┬─────────────────────────────────┘
                             │
                             ▼
    ┌──────────────────────────────────────────────────────────┐
    │              LOAD BALANCER                                │
    │  Round-robin across N replicas                           │
    └──┬────────────┬────────────┬─────────────────────────────┘
       │            │            │
       ▼            ▼            ▼
    ┌───────┐  ┌───────┐  ┌───────┐
    │ Pod 1 │  │ Pod 2 │  │ Pod N │   Kubernetes Deployment
    │ API + │  │ API + │  │ API + │   HPA: scale on CPU
    │ Model │  │ Model │  │ Model │
    └───┬───┘  └───┬───┘  └───┬───┘
        │          │          │
        └──────────┼──────────┘
                   │
    ┌──────────────┴───────────────────────────────────────────┐
    │              SHARED SERVICES                              │
    │  Redis (shared cache)       PostgreSQL (feedback)        │
    │  S3/GCS (model weights)     Prometheus + Grafana         │
    │  ELK/Loki (log aggregation) Model Registry (MLflow)      │
    └──────────────────────────────────────────────────────────┘
```

## Local vs Production Component Mapping

| Component | Local Implementation | Production Replacement |
|---|---|---|
| API Gateway | None (direct access) | Nginx / Traefik / Kong |
| TLS | None (HTTP localhost) | Cert-manager / ACM |
| Auth | None | OAuth2 / JWT with key rotation |
| Rate limiting | slowapi (in-process) | Gateway-level (Kong, Nginx) |
| Scaling | Single process | K8s HPA, 2-10 replicas |
| Cache | In-memory dict per (text, model) | Redis Cluster |
| Feedback store | SQLite | PostgreSQL |
| Model storage | Local filesystem | S3/GCS with versioning |
| Logging | structlog to stdout | ELK / Loki + Grafana dashboards |
| Metrics | None (log-based) | Prometheus + Grafana |
| CI/CD | None | GitHub Actions -> GHCR -> K8s |
| Model serving | PyTorch in-process | ONNX Runtime on x86 (2-3x faster) or Triton |

## Key Design Decisions

**Why DistilBERT as default (not ModernBERT)?**
ModernBERT scores higher (0.917 vs 0.905 F1) but is 2.2x larger (574MB vs 256MB). DistilBERT loads faster, uses less memory, and the 1.2% F1 difference doesn't justify doubling infrastructure cost at scale. All models remain accessible via the API for comparison.

**Why multi-model + ensemble?**
Different models disagree on borderline cases ("The earth orbits the sun" — DistilBERT 91.7% CLAIM, BERT 28.5% NOT CLAIM). The ensemble averages calibrated probabilities, smoothing out individual model biases. The compare endpoint makes disagreements transparent.

**Why PyTorch over ONNX locally?**
Benchmarking showed ONNX Runtime is 0.56x the speed of PyTorch on Apple Silicon MPS. On x86 production servers, ONNX would be preferred (typically 2-3x faster than PyTorch on CPU).

**Why exact-match cache (not normalized)?**
An earlier implementation normalized cache keys to lowercase, causing "GDP grew 2.1%" and "I think GDP grew 2.1%" to share results. The current cache keys on `(exact_text, model_name)` — only identical inputs to the same model return cached results.

**Why temperature scaling for calibration?**
Raw softmax outputs from fine-tuned models are poorly calibrated. Temperature scaling (T=2.15) reduced NLL by 40.9%, making confidence scores meaningful for downstream decision-making.
