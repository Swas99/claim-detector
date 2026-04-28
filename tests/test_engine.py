"""Tests for the inference engine components."""

import pytest

from src.engine.calibration import TemperatureScaler


class TestTemperatureScaler:
    def test_default_temperature(self):
        scaler = TemperatureScaler(temperature=1.0)
        import numpy as np
        logits = np.array([[1.0, 2.0]])
        probs = scaler.calibrate(logits)
        assert probs.shape == (1, 2)
        assert abs(probs.sum() - 1.0) < 1e-6

    def test_higher_temperature_flattens(self):
        import numpy as np
        logits = np.array([[1.0, 3.0]])
        probs_t1 = TemperatureScaler(1.0).calibrate(logits)
        probs_t5 = TemperatureScaler(5.0).calibrate(logits)
        assert probs_t5[0, 1] < probs_t1[0, 1]

    def test_save_and_load(self, tmp_path):
        scaler = TemperatureScaler(temperature=2.5)
        path = tmp_path / "cal.json"
        scaler.save(path)
        loaded = TemperatureScaler.load(path)
        assert abs(loaded.temperature - 2.5) < 1e-6
