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

# Train all models
make train

# Export best model to ONNX
make export-onnx

# Start API
make serve
```

## API Usage

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Empire State Building is the tallest building in New York City."}'

# Response
# {"is_claim": true, "confidence": 0.97, "attribution": [...]}

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["The earth is round.", "I love pizza.", "GDP grew 2.1% last quarter."]}'

# Health check
curl http://localhost:8000/health

# Model info
curl http://localhost:8000/model/info
```

## Docker

```bash
make docker-build
make docker-up
# API available at http://localhost:8000
```

## Testing

```bash
make test           # Unit + integration tests
make load-test      # Locust load test (50 concurrent users, 60s)
```

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full production architecture diagram and local-vs-prod component mapping.

## Models

Four models trained and compared on the same dataset (composite claim detection, 2,600 test samples):

| Model | Params | F1 Score | Accuracy | Training Time | Role |
|---|---|---|---|---|---|
| TF-IDF + XGBoost | ~2MB | 0.797 | 82.0% | <1s | Classical baseline |
| DistilBERT | 66M | 0.905 | 91.1% | 12 min (MPS) | Deployed model |
| BERT-base | 110M | 0.905 | 91.2% | 23 min (MPS) | Paper reproduction |
| ModernBERT | 150M | 0.917 | 92.2% | 34 min (MPS) | Best overall |

Note: DeBERTa-v3 was attempted but has numerical instability (NaN loss) on Apple Silicon MPS. ModernBERT was used as the SOTA comparison instead.

## Project Structure

```
claim-detector/
├── src/
│   ├── data/          # Dataset loading and preprocessing
│   ├── training/      # Model fine-tuning scripts
│   ├── engine/        # Inference pipeline (cascading, cache, calibration)
│   ├── api/           # FastAPI server and routes
│   └── feedback/      # Uncertain prediction logging (SQLite)
├── tests/             # Unit, integration, security, load tests
├── notebooks/         # Training report and analysis
├── docs/              # Architecture, model card, phase plans
├── data/              # Raw dataset (gitignored)
└── models/            # Trained weights (gitignored)
```

## References

- Bell, A. (2025). "Less Can be More: An Empirical Evaluation of Small and Large Language Models for Sentence-level Claim Detection." FEVER Workshop.
- Dataset: [VeritaResearch/claim-extraction](https://github.com/VeritaResearch/claim-extraction)
