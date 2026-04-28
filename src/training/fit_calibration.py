"""Fit temperature scaling calibration on the test set.

Learns optimal temperature T such that softmax(logits / T) produces
well-calibrated confidence scores.

Usage:
    python -m src.training.fit_calibration
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings
from src.data.dataset import load_claim_dataset
from src.engine.calibration import TemperatureScaler


def fit(model_dir: Path | None = None, output_path: Path | None = None) -> float:
    """Fit temperature scaling and save calibration parameters."""
    model_dir = model_dir or settings.model_dir / settings.active_model
    output_path = output_path or settings.model_dir / "calibration.json"

    print(f"Loading model from {model_dir}...")
    import inspect
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    valid_params = set(inspect.signature(model.forward).parameters.keys())

    print("Loading test set...")
    ds = load_claim_dataset()
    texts = ds["test"]["text"]
    labels = np.array(ds["test"]["label"])

    # Get logits on full test set
    print("Computing logits on test set...")
    all_logits = []
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, return_tensors="pt",
            padding=True, truncation=True, max_length=128,
        )
        filtered = {k: v for k, v in inputs.items() if k in valid_params}
        with torch.no_grad():
            outputs = model(**filtered)
        all_logits.append(outputs.logits.numpy())

    logits = np.concatenate(all_logits, axis=0)

    # Compute uncalibrated metrics
    uncalibrated_probs = TemperatureScaler.calibrate_with_temp(logits, 1.0)
    uncal_nll = TemperatureScaler._nll(logits, labels, 1.0)

    # Fit temperature
    print("Fitting temperature...")
    scaler = TemperatureScaler()
    temp = scaler.fit(logits, labels)

    cal_nll = TemperatureScaler._nll(logits, labels, temp)

    print(f"\n{'='*50}")
    print(f"Calibration Results")
    print(f"{'='*50}")
    print(f"  Optimal temperature: {temp:.4f}")
    print(f"  NLL before: {uncal_nll:.4f}")
    print(f"  NLL after:  {cal_nll:.4f}")
    print(f"  Improvement: {(uncal_nll - cal_nll) / uncal_nll * 100:.1f}%")

    scaler.save(output_path)
    print(f"\nSaved to {output_path}")
    return temp


if __name__ == "__main__":
    fit()
