"""ClaimDetector: orchestrates the inference pipeline.

Supports multiple models and ensemble prediction.

Pipeline:
    1. Model inference (single model or ensemble of all transformers)
    2. Confidence calibration (temperature scaling)
    3. Token attribution (attention weights) — single model only
"""

import inspect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings
from src.engine.attribution import AttentionAttributor
from src.engine.calibration import TemperatureScaler


@dataclass
class Prediction:
    """Single prediction result."""
    is_claim: bool
    confidence: float
    model: str = "distilbert-base-uncased"
    attribution: list[dict] | None = None


@dataclass
class ComparisonResult:
    """All models' predictions for one sentence."""
    text: str
    predictions: dict[str, Prediction]
    ensemble: Prediction


TRANSFORMER_MODELS = {
    "distilbert-base-uncased": {"display": "DistilBERT", "hf_name": "distilbert-base-uncased"},
    "bert-base-uncased": {"display": "BERT", "hf_name": "bert-base-uncased"},
    "ModernBERT-base": {"display": "ModernBERT", "hf_name": "answerdotai/ModernBERT-base"},
}


class ClaimDetector:
    """Multi-model claim detection engine."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or settings.model_dir
        self._models: dict[str, dict] = {}
        self._attributors: dict[str, AttentionAttributor] = {}

        # Load calibration
        cal_path = self.model_dir / "calibration.json"
        if cal_path.exists():
            self.calibrator = TemperatureScaler.load(cal_path)
        else:
            self.calibrator = TemperatureScaler(temperature=1.0)

        # Load the default model eagerly
        self._load_model(settings.active_model)

    def _load_model(self, model_name: str) -> None:
        """Load a model if not already loaded."""
        if model_name in self._models:
            return

        model_path = self.model_dir / model_name
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load tokenizer — fall back to HuggingFace hub name if local config is broken
        hf_name = TRANSFORMER_MODELS.get(model_name, {}).get("hf_name")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        except (ValueError, OSError):
            if hf_name:
                tokenizer = AutoTokenizer.from_pretrained(hf_name)
            else:
                raise
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()

        valid_params = set(inspect.signature(model.forward).parameters.keys())

        self._models[model_name] = {
            "model": model,
            "tokenizer": tokenizer,
            "valid_params": valid_params,
        }
        self._attributors[model_name] = AttentionAttributor(model, tokenizer)

    def _infer(self, text: str, model_name: str) -> np.ndarray:
        """Run inference on a single text, return raw logits."""
        self._load_model(model_name)
        m = self._models[model_name]

        inputs = m["tokenizer"](
            text, return_tensors="pt", padding=True,
            truncation=True, max_length=settings.max_seq_length,
        )
        filtered = {k: v for k, v in inputs.items() if k in m["valid_params"]}

        with torch.no_grad():
            outputs = m["model"](**filtered)
        return outputs.logits.numpy()[0]

    def predict(self, text: str, model_name: str | None = None) -> Prediction:
        """Predict with a single model."""
        model_name = model_name or settings.active_model
        logits = self._infer(text, model_name)
        probs = self.calibrator.calibrate(logits.reshape(1, -1))[0]
        claim_prob = float(probs[1])

        attr = self._attributors.get(model_name)
        attribution = attr.attribute(text) if attr else None

        return Prediction(
            is_claim=claim_prob >= settings.confidence_threshold,
            confidence=claim_prob,
            model=model_name,
            attribution=attribution,
        )

    def predict_ensemble(self, text: str) -> Prediction:
        """Ensemble: average calibrated probabilities from all transformer models."""
        probs_list = []
        for model_name in TRANSFORMER_MODELS:
            try:
                logits = self._infer(text, model_name)
                probs = self.calibrator.calibrate(logits.reshape(1, -1))[0]
                probs_list.append(probs[1])
            except (FileNotFoundError, Exception):
                continue

        if not probs_list:
            return self.predict(text)

        avg_prob = float(np.mean(probs_list))
        return Prediction(
            is_claim=avg_prob >= settings.confidence_threshold,
            confidence=avg_prob,
            model="ensemble",
        )

    def compare(self, text: str) -> ComparisonResult:
        """Run all models and return side-by-side comparison."""
        predictions = {}
        for model_name, info in TRANSFORMER_MODELS.items():
            try:
                predictions[info["display"]] = self.predict(text, model_name)
            except Exception:
                continue

        ensemble = self.predict_ensemble(text)
        return ComparisonResult(text=text, predictions=predictions, ensemble=ensemble)

    @property
    def available_models(self) -> list[str]:
        """List all available model names."""
        available = []
        for model_name in TRANSFORMER_MODELS:
            if (self.model_dir / model_name).exists():
                available.append(model_name)
        available.append("ensemble")
        return available

    @property
    def info(self) -> dict:
        return {
            "default_model": settings.active_model,
            "available_models": self.available_models,
            "calibration_temperature": self.calibrator.temperature,
        }
