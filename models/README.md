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
├── deberta-v3-base/
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── config.json
│   └── metrics.json
└── onnx/
    ├── model.onnx
    └── tokenizer.json
```

Recreate with `make train && make export-onnx`.
