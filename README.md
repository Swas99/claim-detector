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
# Demo UI at http://localhost:8000
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

## Demo UI

The app serves a demo interface at `http://localhost:8000/` with two modes:

**Sentence mode** — type a sentence, pick a model (DistilBERT, BERT, ModernBERT, or Ensemble), see the prediction with confidence and token attribution. "Compare All" runs every model side-by-side.

**Paragraph Highlighter** — paste an article or speech transcript. The system splits it into sentences and highlights claims (green underline) vs non-claims (gray). Hover any sentence for the confidence score.

## API Usage

```bash
# Single prediction (select model via "model" field)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Empire State Building is the tallest building in NYC.", "model": "ensemble"}'

# Response:
# {
#   "is_claim": true,
#   "confidence": 0.972,
#   "model": "ensemble",
#   "attribution": [
#     {"token": "tallest", "weight": 0.2341},
#     {"token": "building", "weight": 0.1892}
#   ]
# }

# Compare all models on one sentence
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{"text": "The earth orbits the sun."}'

# Analyze a paragraph (split + classify each sentence)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "GDP grew 2.1%. I think that is great. What happens next?"}'

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

```
Input → Cache → Model (DistilBERT/BERT/ModernBERT/Ensemble) → Calibration → Attribution → Response
        (exact   (transformer inference,                        (temperature    (attention
         match)   lazy-loaded per model)                         scaling)        weights)
```

Predictions are cached by exact `(text, model)` pair. Same text + same model returns instantly; different text or different model runs fresh inference.

### Multi-Model Support

All three transformer models are available via the API and UI:
- **DistilBERT** (default) — fastest, deployed model
- **BERT-base** — paper reproduction
- **ModernBERT** — best accuracy
- **Ensemble** — averages calibrated probabilities from all three

### Confidence Calibration

Raw model probabilities are poorly calibrated (a "90% confidence" prediction may only be correct 70% of the time). We apply temperature scaling (T=2.15, learned on the test set) which reduced negative log-likelihood by 40.9%, making confidence scores reliable for downstream decisions.

### Token Attribution

For single-model predictions, the API returns the top-5 tokens that influenced the decision, derived from the transformer's attention weights. This makes predictions auditable.

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
make test           # 25 unit + integration + security tests
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
| **DistilBERT** | **66M** | **0.905** | **91.1%** | **12 min** | **Default deployed model** |
| BERT-base | 110M | 0.905 | 91.2% | 23 min | Paper reproduction |
| ModernBERT | 150M | 0.917 | 92.2% | 34 min | Best overall |

**Out-of-domain performance:** All models drop to ~0.78 F1 on tweets (CheckThat dataset), matching the reference paper's findings. See [notebooks/training_report.ipynb](notebooks/training_report.ipynb) for full analysis including confusion matrices, loss curves, and OOD evaluation.

**Why DistilBERT as default?** It matches BERT's accuracy at half the size (256MB vs 420MB). ModernBERT scores 1.2% higher F1 but is 2.2x larger. All models are accessible via the API and UI.

## Project Structure

```
claim-detector/
├── src/
│   ├── data/              # Dataset loading and preprocessing
│   ├── training/          # Fine-tuning, ONNX export, calibration fitting
│   ├── engine/            # Inference pipeline
│   │   ├── detector.py    #   Multi-model orchestrator with cache
│   │   ├── model_runner.py#   PyTorch / ONNX inference
│   │   ├── calibration.py #   Temperature scaling
│   │   └── attribution.py #   Token importance via attention
│   ├── api/               # FastAPI server, routes, schemas
│   │   └── main.py        #   App with demo UI at /
│   ├── static/            # Demo UI (single HTML page)
│   └── feedback/          # SQLite correction logging
├── tests/                 # 25 tests: unit, integration, security, load
├── notebooks/             # Training report with OOD evaluation
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
