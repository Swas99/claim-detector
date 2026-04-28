# Phase 2: Model Training + Comparison

**Status:** IN PROGRESS
**Estimated time:** ~3 hours

## Models

1. **TF-IDF + XGBoost** — classical ML baseline
2. **DistilBERT** (distilbert-base-uncased, 66M) — lightweight primary candidate
3. **BERT** (bert-base-uncased, 110M) — paper reproduction
4. **DeBERTa-v3** (microsoft/deberta-v3-base, 86M) — SOTA comparison

## Training Config (Transformers)

- Epochs: 5
- Batch size: 16
- Learning rate: 2e-5
- Weight decay: 0.01
- Max sequence length: 128
- Device: MPS (Apple Silicon)
- Eval strategy: each epoch
- Save: best checkpoint by eval F1

## Tasks

- [ ] Install full ML dependencies
- [ ] Write src/training/train_tfidf.py
- [ ] Train TF-IDF + XGBoost, save metrics
- [ ] Write src/training/train_transformer.py (generic for all 3)
- [ ] Train DistilBERT, save metrics
- [ ] Train BERT-base, save metrics
- [ ] Train DeBERTa-v3-base, save metrics
- [ ] Print comparison table
- [ ] Commit all training code + metrics

## Expected Output

Each model saves to `models/{model_name}/`:
- Model weights (safetensors or pkl)
- metrics.json (accuracy, precision, recall, F1, confusion matrix)
- Training args / config

## Deliverable

```bash
python -m src.training.train_tfidf        # ~5 seconds
python -m src.training.train_transformer --model distilbert-base-uncased  # ~15 min
python -m src.training.train_transformer --model bert-base-uncased        # ~30 min
python -m src.training.train_transformer --model microsoft/deberta-v3-base # ~25 min
```
