# System Architecture

## Overview

The claim detector is a sentence-level classification system that identifies whether a natural language sentence contains a verifiable factual claim. It uses a fine-tuned DistilBERT model behind a cascading inference pipeline with caching, rule-based pre-filtering, and confidence calibration.

## Local Architecture

```
    Client (curl / browser / test suite)
        │
        ▼
┌───────────────────────────────────────────────────┐
│                  FastAPI Server                     │
│  - CORS middleware                                  │
│  - Rate limiting (slowapi, 60 req/min per IP)      │
│  - Structured JSON logging (structlog)             │
│  - Pydantic input validation                       │
│                                                     │
│  Endpoints:                                         │
│    POST /predict        single sentence             │
│    POST /predict/batch  up to 100 sentences         │
│    GET  /health         liveness check              │
│    GET  /model/info     metadata + eval metrics     │
│    POST /feedback       correction logging          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│              Inference Pipeline                     │
│                                                     │
│  1. CACHE LOOKUP ──► hit? return immediately        │
│     │ miss                                          │
│     ▼                                               │
│  2. FAST FILTER ──► question? opinion? → skip model │
│     │ uncertain                                     │
│     ▼                                               │
│  3. MODEL INFERENCE (DistilBERT / ONNX)            │
│     │                                               │
│     ▼                                               │
│  4. CONFIDENCE CALIBRATION (T=2.15)                │
│     │                                               │
│     ▼                                               │
│  5. TOKEN ATTRIBUTION (attention weights)          │
│     │                                               │
│     ▼                                               │
│  6. CACHE STORE + RETURN                           │
└───────────────────────────────────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────────────┐
│              Persistence                            │
│  - Feedback store: SQLite (corrections)            │
│  - Cache: in-memory LRU (10k entries, 1hr TTL)    │
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
| Cache | In-memory LRU (cachetools) | Redis Cluster |
| Feedback store | SQLite | PostgreSQL |
| Model storage | Local filesystem | S3/GCS with versioning |
| Logging | structlog to stdout | ELK / Loki + Grafana dashboards |
| Metrics | None (log-based) | Prometheus + Grafana |
| CI/CD | None | GitHub Actions -> GHCR -> K8s |
| Model serving | PyTorch in-process | ONNX Runtime on x86 (2-3x faster) or Triton |

## Key Design Decisions

**Why DistilBERT for deployment (not ModernBERT)?**
ModernBERT scores higher (0.917 vs 0.905 F1) but is 2.2x larger (574MB vs 256MB). DistilBERT loads faster, uses less memory, and the 1.2% F1 difference doesn't justify doubling infrastructure cost at scale.

**Why PyTorch over ONNX locally?**
Benchmarking showed ONNX Runtime is 0.56x the speed of PyTorch on Apple Silicon MPS. On x86 production servers, ONNX would be preferred (typically 2-3x faster than PyTorch on CPU).

**Why a cascading pipeline?**
The fast filter catches ~30-40% of inputs (questions, opinions, greetings) without any model inference. At scale, this cuts compute cost significantly. The cache further reduces redundant inference.

**Why temperature scaling for calibration?**
Raw softmax outputs from fine-tuned models are poorly calibrated. Temperature scaling (T=2.15) reduced NLL by 40.9%, making confidence scores meaningful for downstream decision-making.
