# Claim Detector

Sentence-level claim detection API. Given a natural language sentence, determines whether it contains a verifiable factual claim.

> **Claim detection** is not fact-checking. It identifies whether a sentence *can be* fact-checked — the triage step upstream of verification.

## Quick Start

```bash
# Setup
make setup
source .venv/bin/activate

# Download dataset
make download-data

# Train all models (TF-IDF ~1s, DistilBERT ~12m, BERT ~23m, ModernBERT ~34m)
make train

# Fit confidence calibration
python -m src.training.fit_calibration

# Start API
make serve
# API docs at http://localhost:8000/docs
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Empire State Building is the tallest building in New York City."}'

# Response:
# {
#   "is_claim": true,
#   "confidence": 0.972,
#   "source": "model",
#   "cached": false,
#   "attribution": [
#     {"token": "tallest", "weight": 0.2341},
#     {"token": "building", "weight": 0.1892},
#     ...
#   ]
# }

# Batch prediction (up to 100 sentences)
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The earth is round.", "I love pizza.", "GDP grew 2.1%."]}'

# Health check
curl http://localhost:8000/health

# Model info + evaluation metrics
curl http://localhost:8000/model/info

# Submit correction (for model improvement)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "predicted_is_claim": true, "correct_is_claim": false}'
```

## Inference Pipeline

Requests pass through a cascading pipeline that skips unnecessary computation:

```
Input → Cache Lookup → Fast Filter → Model (DistilBERT) → Calibration → Attribution → Response
         (exact match)  (rules for     (transformer        (T=2.15,       (attention
          LRU, 1hr TTL)  questions,      inference)          40.9% NLL      weights,
                         opinions)                           improvement)   top-5 tokens)
```

The fast filter catches ~30-40% of inputs (questions, opinions, greetings) without any model inference. The cache eliminates redundant computation for repeated queries.

## Docker

```bash
make docker-build
make docker-up
# API available at http://localhost:8000
```

## Testing

```bash
make test           # 38 unit + integration + security tests
make load-test      # Locust: 50 concurrent users, 60s
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full production architecture diagram and local-vs-prod component mapping.

See [docs/model_card.md](docs/model_card.md) for model details, limitations, and ethical considerations.

## Models

Four models trained and compared on the same dataset (composite claim detection, 2,600 test samples):

| Model | Params | F1 Score | Accuracy | Training Time | Role |
|---|---|---|---|---|---|
| TF-IDF + XGBoost | ~2MB | 0.797 | 82.0% | <1s | Classical baseline |
| **DistilBERT** | **66M** | **0.905** | **91.1%** | **12 min** | **Deployed model** |
| BERT-base | 110M | 0.905 | 91.2% | 23 min | Paper reproduction |
| ModernBERT | 150M | 0.917 | 92.2% | 34 min | Best overall |

DistilBERT was chosen for deployment: nearly matches BERT's accuracy at half the size and training time. ModernBERT is the best overall but 2.2x larger.

Note: DeBERTa-v3 was attempted but has numerical instability (NaN loss) on Apple Silicon MPS. ModernBERT was used as the SOTA comparison instead.

## Project Structure

```
claim-detector/
├── src/
│   ├── data/              # Dataset loading and preprocessing
│   ├── training/          # Fine-tuning, ONNX export, calibration fitting
│   ├── engine/            # Inference pipeline
│   │   ├── detector.py    #   Orchestrator (cache → filter → model → calibrate)
│   │   ├── fast_filter.py #   Rule-based pre-filter
│   │   ├── model_runner.py#   PyTorch / ONNX inference
│   │   ├── cache.py       #   LRU + TTL prediction cache
│   │   ├── calibration.py #   Temperature scaling
│   │   └── attribution.py #   Token importance via attention
│   ├── api/               # FastAPI server, routes, schemas
│   └── feedback/          # SQLite correction logging
├── tests/                 # 38 tests: unit, integration, security, load
├── docs/                  # Architecture diagram, model card
├── Dockerfile             # Multi-stage (builder + slim runtime)
├── docker-compose.yml
├── Makefile               # train / serve / test / docker targets
└── pyproject.toml
```

## References

- Bell, A. (2025). "Less Can be More: An Empirical Evaluation of Small and Large Language Models for Sentence-level Claim Detection." FEVER Workshop.
- Dataset: [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
