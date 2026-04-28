"""Tests for the inference engine components."""

import pytest

from src.engine.cache import PredictionCache
from src.engine.calibration import TemperatureScaler
from src.engine.fast_filter import fast_filter


class TestFastFilter:
    def test_question_not_claim(self):
        result = fast_filter("What time does the meeting start?")
        assert result.decision is False
        assert result.rule == "question"

    def test_opinion_not_claim(self):
        result = fast_filter("I think pizza is the best food ever.")
        assert result.decision is False
        assert result.rule == "opinion_hedge"

    def test_greeting_not_claim(self):
        result = fast_filter("Hello, how are you?")
        assert result.decision is False

    def test_imperative_not_claim(self):
        result = fast_filter("Stop doing that right now.")
        assert result.decision is False
        assert result.rule == "imperative"

    def test_statistic_is_claim(self):
        result = fast_filter("Sales increased by 15% last quarter.")
        assert result.decision is True
        assert result.rule == "statistic_pattern"

    def test_short_text_not_claim(self):
        result = fast_filter("Hi")
        assert result.decision is False
        assert result.rule == "too_short"

    def test_ambiguous_returns_none(self):
        result = fast_filter("The Empire State Building is in New York City.")
        assert result.decision is None

    def test_empty_after_strip(self):
        result = fast_filter("   ")
        assert result.decision is False


class TestPredictionCache:
    def test_put_and_get(self):
        cache = PredictionCache(max_size=100, ttl=60)
        cache.put("test sentence", {"is_claim": True, "confidence": 0.95})
        result = cache.get("test sentence")
        assert result is not None
        assert result["is_claim"] is True
        assert result["cached"] is True

    def test_miss_returns_none(self):
        cache = PredictionCache(max_size=100, ttl=60)
        assert cache.get("nonexistent") is None

    def test_case_insensitive(self):
        cache = PredictionCache(max_size=100, ttl=60)
        cache.put("Test Sentence", {"is_claim": True, "confidence": 0.9})
        result = cache.get("test sentence")
        assert result is not None

    def test_stats(self):
        cache = PredictionCache(max_size=100, ttl=60)
        cache.put("a", {"is_claim": True, "confidence": 0.9})
        cache.get("a")  # hit
        cache.get("b")  # miss
        stats = cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_clear(self):
        cache = PredictionCache(max_size=100, ttl=60)
        cache.put("a", {"is_claim": True, "confidence": 0.9})
        cache.clear()
        assert cache.stats["size"] == 0


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
        # Higher temperature -> more uniform distribution
        assert probs_t5[0, 1] < probs_t1[0, 1]

    def test_save_and_load(self, tmp_path):
        scaler = TemperatureScaler(temperature=2.5)
        path = tmp_path / "cal.json"
        scaler.save(path)
        loaded = TemperatureScaler.load(path)
        assert abs(loaded.temperature - 2.5) < 1e-6
