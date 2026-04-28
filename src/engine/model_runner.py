"""Model inference runner supporting PyTorch and ONNX backends.

Loads a trained model and runs inference on text inputs.
Returns raw logits — calibration and thresholding happen downstream.
"""

import inspect
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import settings


class ModelRunner:
    """Runs transformer inference on text inputs."""

    def __init__(
        self,
        model_dir: Path | None = None,
        use_onnx: bool | None = None,
        max_length: int | None = None,
    ):
        self.max_length = max_length or settings.max_seq_length
        model_dir = model_dir or settings.model_dir / settings.active_model
        use_onnx = use_onnx if use_onnx is not None else settings.use_onnx

        if use_onnx:
            self._load_onnx(model_dir)
        else:
            self._load_pytorch(model_dir)

    def _load_pytorch(self, model_dir: Path) -> None:
        """Load PyTorch model."""
        self.backend = "pytorch"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

        # Determine valid input params for this model
        self._valid_params = set(
            inspect.signature(self.model.forward).parameters.keys()
        )

    def _load_onnx(self, model_dir: Path) -> None:
        """Load ONNX model via Optimum."""
        self.backend = "onnx"
        onnx_dir = settings.model_dir / "onnx"

        if not (onnx_dir / "model.onnx").exists():
            # Fall back to PyTorch
            print(f"ONNX model not found at {onnx_dir}, falling back to PyTorch")
            self._load_pytorch(model_dir)
            return

        from optimum.onnxruntime import ORTModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.model = ORTModelForSequenceClassification.from_pretrained(str(onnx_dir))
        self._valid_params = set(
            inspect.signature(self.model.forward).parameters.keys()
        )

    def predict(self, texts: list[str]) -> np.ndarray:
        """Run inference on a list of texts.

        Returns:
            np.ndarray of shape (batch_size, 2) with logits for [not_claim, claim].
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )

        # Filter to valid params
        inputs = {k: v for k, v in inputs.items() if k in self._valid_params}

        if self.backend == "pytorch":
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits.numpy()
        else:
            outputs = self.model(**inputs)
            return outputs.logits.detach().numpy()

    def predict_single(self, text: str) -> np.ndarray:
        """Run inference on a single text. Returns logits of shape (2,)."""
        return self.predict([text])[0]

    @property
    def model_name(self) -> str:
        return settings.active_model

    @property
    def info(self) -> dict:
        return {
            "model_name": self.model_name,
            "backend": self.backend,
            "max_length": self.max_length,
        }
