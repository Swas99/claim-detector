# Model Artifacts

Trained model weights are stored here but excluded from git (large files).

## Expected contents after training

```
models/
├── tfidf-xgboost/
│   ├── vectorizer.pkl
│   ├── model.pkl
│   └── metrics.json
├── distilbert-base-uncased/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── metrics.json
├── bert-base-uncased/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── metrics.json
├── ModernBERT-base/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── metrics.json
├── calibration.json         (temperature scaling parameter)
└── ood_metrics.json         (out-of-domain evaluation results)
```

Recreate with `make train`.
