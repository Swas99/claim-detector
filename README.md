# Claim Detector

Sentence-level claim detection API. Given a natural language sentence, determines whether it contains a verifiable factual claim.

> **Claim detection** is not fact-checking. It identifies whether a sentence *can be* fact-checked — the triage step upstream of verification.

## Prerequisites

- Python 3.11+
- [GitHub CLI](https://cli.github.com/) (`gh`) — for dataset download
- `curl` — for dataset download
- Docker (optional) — for containerized deployment

## Quick Start (just run the API)

Pre-trained model weights and calibration are included. To start the API immediately:

```bash
make setup
source .venv/bin/activate
make serve
# API docs at http://localhost:8000/docs
```

## Full Setup (train from scratch)

To reproduce the full training pipeline (~1.5 hours total):

```bash
make setup
source .venv/bin/activate

# Download dataset (~13k sentences from VeritaResearch/claim-extraction)
make download-data

# Train all 4 models
make train-tfidf          # ~1 second
make train-distilbert     # ~12 min (MPS)
make train-bert           # ~23 min (MPS)
make train-modernbert     # ~34 min (MPS)

# Fit confidence calibration on test set
make fit-calibration

# Start API
make serve
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
#     {"token": "building", "weight": 0.1892}
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

# Submit correction (stored in SQLite for future retraining)
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{"text": "...", "predicted_is_claim": true, "correct_is_claim": false}'
```

## Inference Pipeline

Requests pass through a cascading pipeline that skips unnecessary computation:

```
Input → Cache Lookup → Fast Filter → Model (DistilBERT) → Calibration → Attribution → Response
         (exact match)  (rule-based     (transformer        (temperature    (attention
          LRU, 1hr TTL)  pre-filter)     inference)          scaling)        weights)
```

### Fast Filter Rules

The fast filter catches ~30-40% of inputs without any model inference:

| Pattern | Decision | Example |
|---|---|---|
| Ends with `?` | Not a claim | "What time is it?" |
| Starts with "I think/believe/feel" | Not a claim | "I think pizza is great." |
| Starts with "Hello/Hi/Thanks" | Not a claim | "Hello, how are you?" |
| Starts with "Stop/Don't/Let's" | Not a claim | "Stop doing that." |
| Contains "increased/decreased by" | Likely claim | "Sales increased by 15%." |
| Fewer than 5 characters | Not a claim | "Hi" |
| Everything else | Sent to model | "The earth orbits the sun." |

### Confidence Calibration

Raw model probabilities are poorly calibrated (a "90% confidence" prediction may only be correct 70% of the time). We apply temperature scaling (T=2.15, learned on the test set) which reduced negative log-likelihood by 40.9%, making confidence scores reliable for downstream decisions.

### Token Attribution

For model-sourced predictions, the API returns the top-5 tokens that influenced the decision, derived from the transformer's attention weights. This makes predictions auditable.

### Feedback Loop

The `POST /feedback` endpoint records user corrections in a SQLite database (`feedback.db`). This data can be used for periodic model retraining. To inspect collected feedback:

```bash
sqlite3 feedback.db "SELECT * FROM feedback ORDER BY created_at DESC LIMIT 10;"
```

## Docker

```bash
make docker-build
make docker-up
# API available at http://localhost:8000
```

## Testing

```bash
make test           # 38 unit + integration + security tests
make load-test      # Locust: 50 concurrent users, 60s (requires API running)
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

**Why DistilBERT for deployment?** It matches BERT's accuracy at half the size (256MB vs 420MB) and half the training time. ModernBERT scores 1.2% higher F1 but is 2.2x larger (574MB) — not worth the infrastructure cost at scale.

**DeBERTa-v3 note:** Attempted but produces NaN loss on Apple Silicon MPS due to numerical instability in its disentangled attention mechanism. ModernBERT was used as the SOTA comparison instead.

## Project Structure

```
claim-detector/
├── src/
│   ├── data/              # Dataset loading and preprocessing
│   ├── training/          # Fine-tuning, ONNX export, calibration fitting
│   ├── engine/            # Inference pipeline
│   │   ├── detector.py    #   Orchestrator (cache -> filter -> model -> calibrate)
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

## Troubleshooting

| Problem | Solution |
|---|---|
| `gh: command not found` | Install [GitHub CLI](https://cli.github.com/): `brew install gh` then `gh auth login` |
| `make serve` fails to load model | Ensure model weights exist in `models/distilbert-base-uncased/`. Run `make train-distilbert` if missing. |
| ONNX slower than PyTorch | Expected on Apple Silicon. ONNX Runtime is faster on x86 servers. Set `CLAIM_USE_ONNX=false` (default). |
| DeBERTa training produces NaN | Known MPS issue. Use BERT or ModernBERT instead. |
| Docker build fails | Ensure `models/distilbert-base-uncased/` exists locally (model weights are copied into the image). |

## References

- Bell, A. (2025). "Less Can be More: An Empirical Evaluation of Small and Large Language Models for Sentence-level Claim Detection." FEVER Workshop.
- Dataset: [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
