"""ClaimDetector: orchestrates the full inference pipeline.

Pipeline:
    1. Cache lookup (exact match)
    2. Fast filter (rule-based skip for obvious cases)
    3. Model inference (transformer)
    4. Confidence calibration (temperature scaling)
    5. Cache store
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import settings
from src.engine.attribution import AttentionAttributor
from src.engine.cache import PredictionCache
from src.engine.calibration import TemperatureScaler
from src.engine.fast_filter import fast_filter
from src.engine.model_runner import ModelRunner


@dataclass
class Prediction:
    """Single prediction result."""
    is_claim: bool
    confidence: float
    source: str = "model"  # "model", "fast_filter", "cache"
    cached: bool = False
    filter_rule: str | None = None
    attribution: list[dict] | None = None


class ClaimDetector:
    """Main claim detection engine.

    Orchestrates cache, fast filter, model inference, and calibration.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        use_onnx: bool | None = None,
        enable_cache: bool = True,
        enable_fast_filter: bool = True,
    ):
        self.runner = ModelRunner(model_dir=model_dir, use_onnx=use_onnx)
        self.cache = PredictionCache() if enable_cache else None
        self.enable_fast_filter = enable_fast_filter

        # Load calibration if available
        cal_path = settings.model_dir / "calibration.json"
        if cal_path.exists():
            self.calibrator = TemperatureScaler.load(cal_path)
        else:
            self.calibrator = TemperatureScaler(temperature=1.0)

        # Attribution (only available with PyTorch backend)
        if self.runner.backend == "pytorch":
            self.attributor = AttentionAttributor(self.runner.model, self.runner.tokenizer)
        else:
            self.attributor = None

    def predict(self, text: str) -> Prediction:
        """Run the full prediction pipeline on a single sentence."""

        # 1. Cache lookup
        if self.cache is not None:
            cached = self.cache.get(text)
            if cached is not None:
                return Prediction(
                    is_claim=cached["is_claim"],
                    confidence=cached["confidence"],
                    source="cache",
                    cached=True,
                )

        # 2. Fast filter
        if self.enable_fast_filter:
            ff_result = fast_filter(text)
            if ff_result.decision is not None:
                pred = Prediction(
                    is_claim=ff_result.decision,
                    confidence=ff_result.confidence,
                    source="fast_filter",
                    filter_rule=ff_result.rule,
                )
                if self.cache is not None:
                    self.cache.put(text, {
                        "is_claim": pred.is_claim,
                        "confidence": pred.confidence,
                    })
                return pred

        # 3. Model inference
        logits = self.runner.predict_single(text)

        # 4. Calibration
        probs = self.calibrator.calibrate(logits.reshape(1, -1))[0]
        claim_prob = float(probs[1])

        is_claim = claim_prob >= settings.confidence_threshold

        # 4b. Attribution (token importance)
        attr = None
        if self.attributor is not None:
            attr = self.attributor.attribute(text)

        pred = Prediction(
            is_claim=is_claim,
            confidence=claim_prob,
            source="model",
            attribution=attr,
        )

        # 5. Cache store
        if self.cache is not None:
            self.cache.put(text, {
                "is_claim": pred.is_claim,
                "confidence": pred.confidence,
            })

        return pred

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        """Run prediction on multiple sentences.

        Uses the fast filter for obvious cases and batches the rest
        through the model for efficiency.
        """
        results: list[Prediction] = []
        model_indices: list[int] = []
        model_texts: list[str] = []

        # First pass: resolve from cache/filter, collect model-needed indices
        for i, text in enumerate(texts):
            resolved = False

            # Cache lookup
            if self.cache is not None:
                cached = self.cache.get(text)
                if cached is not None:
                    results.append(Prediction(
                        is_claim=cached["is_claim"],
                        confidence=cached["confidence"],
                        source="cache",
                        cached=True,
                    ))
                    resolved = True

            # Fast filter
            if not resolved and self.enable_fast_filter:
                ff_result = fast_filter(text)
                if ff_result.decision is not None:
                    pred = Prediction(
                        is_claim=ff_result.decision,
                        confidence=ff_result.confidence,
                        source="fast_filter",
                        filter_rule=ff_result.rule,
                    )
                    results.append(pred)
                    if self.cache is not None:
                        self.cache.put(text, {
                            "is_claim": pred.is_claim,
                            "confidence": pred.confidence,
                        })
                    resolved = True

            if not resolved:
                # Placeholder — will be replaced after model inference
                model_indices.append(i)
                model_texts.append(text)
                results.append(Prediction(is_claim=False, confidence=0.0, source="pending"))

        # Second pass: batch model inference for unresolved
        if model_texts:
            logits = self.runner.predict(model_texts)
            probs = self.calibrator.calibrate(logits)

            for j, idx in enumerate(model_indices):
                claim_prob = float(probs[j, 1])
                pred = Prediction(
                    is_claim=claim_prob >= settings.confidence_threshold,
                    confidence=claim_prob,
                    source="model",
                )
                results[idx] = pred
                if self.cache is not None:
                    self.cache.put(model_texts[j], {
                        "is_claim": pred.is_claim,
                        "confidence": pred.confidence,
                    })

        return results

    @property
    def info(self) -> dict:
        """Return model and pipeline info."""
        return {
            **self.runner.info,
            "calibration_temperature": self.calibrator.temperature,
            "cache_enabled": self.cache is not None,
            "fast_filter_enabled": self.enable_fast_filter,
            "cache_stats": self.cache.stats if self.cache else None,
        }
