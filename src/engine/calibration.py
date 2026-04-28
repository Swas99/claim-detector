"""Confidence calibration for claim detection.

Raw softmax probabilities are often poorly calibrated — a model saying
"90% confident" might only be right 70% of the time. Temperature scaling
(Platt scaling) learns a single parameter T to fix this:

    calibrated_prob = softmax(logits / T)

T is learned on a held-out calibration set to minimize negative log-likelihood.
"""

import json
from pathlib import Path

import numpy as np


class TemperatureScaler:
    """Post-hoc temperature scaling for confidence calibration."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits, return calibrated probabilities."""
        scaled = logits / self.temperature
        # Softmax
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    def fit(self, logits: np.ndarray, labels: np.ndarray, lr: float = 0.01, max_iter: int = 1000) -> float:
        """Learn optimal temperature on calibration data.

        Uses simple gradient descent on negative log-likelihood.

        Args:
            logits: Raw model logits, shape (n_samples, 2)
            labels: True labels, shape (n_samples,)
            lr: Learning rate
            max_iter: Max optimization steps

        Returns:
            Optimal temperature value.
        """
        temp = 1.5  # Start slightly above 1

        for _ in range(max_iter):
            probs = self.calibrate_with_temp(logits, temp)
            # NLL gradient w.r.t. temperature
            correct_probs = probs[np.arange(len(labels)), labels]
            correct_probs = np.clip(correct_probs, 1e-10, 1.0)

            # Numerical gradient
            eps = 0.01
            nll_plus = self._nll(logits, labels, temp + eps)
            nll_minus = self._nll(logits, labels, temp - eps)
            grad = (nll_plus - nll_minus) / (2 * eps)

            temp -= lr * grad
            temp = max(0.1, min(10.0, temp))  # Clamp

        self.temperature = temp
        return temp

    @staticmethod
    def calibrate_with_temp(logits: np.ndarray, temperature: float) -> np.ndarray:
        """Softmax with given temperature."""
        scaled = logits / temperature
        exp_scaled = np.exp(scaled - np.max(scaled, axis=-1, keepdims=True))
        return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    @staticmethod
    def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
        """Negative log-likelihood."""
        probs = TemperatureScaler.calibrate_with_temp(logits, temperature)
        correct_probs = probs[np.arange(len(labels)), labels]
        return -np.mean(np.log(np.clip(correct_probs, 1e-10, 1.0)))

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump({"temperature": self.temperature}, f)

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        with open(path) as f:
            data = json.load(f)
        return cls(temperature=data["temperature"])
