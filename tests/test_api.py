"""Integration tests for the API endpoints."""

import os

import pytest

# Use PyTorch backend for tests (faster startup, no ONNX needed)
os.environ["CLAIM_USE_ONNX"] = "false"


class TestPredict:
    def test_claim_sentence(self, client):
        resp = client.post("/predict", json={
            "text": "The Empire State Building is the tallest building in New York City."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_claim"] is True
        assert data["confidence"] > 0.5
        assert data["source"] in ("model", "fast_filter", "cache")

    def test_non_claim_opinion(self, client):
        resp = client.post("/predict", json={
            "text": "I think chocolate ice cream is the best."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_claim"] is False

    def test_non_claim_question(self, client):
        resp = client.post("/predict", json={
            "text": "What is the capital of France?"
        })
        assert resp.status_code == 200
        assert resp.json()["is_claim"] is False

    def test_attribution_present_for_model(self, client):
        resp = client.post("/predict", json={
            "text": "GDP grew by 3.2% in the last fiscal year."
        })
        data = resp.json()
        if data["source"] == "model":
            assert data["attribution"] is not None
            assert len(data["attribution"]) > 0
            assert "token" in data["attribution"][0]
            assert "weight" in data["attribution"][0]

    def test_cache_hit(self, client):
        text = "The population of Tokyo is over 13 million."
        client.post("/predict", json={"text": text})
        resp = client.post("/predict", json={"text": text})
        assert resp.json()["cached"] is True

    def test_response_schema(self, client):
        resp = client.post("/predict", json={"text": "Test sentence."})
        data = resp.json()
        assert "is_claim" in data
        assert "confidence" in data
        assert "source" in data
        assert "cached" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0


class TestBatchPredict:
    def test_batch_multiple(self, client):
        resp = client.post("/predict/batch", json={
            "texts": [
                "The earth is round.",
                "I love pizza.",
                "What time is it?",
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3

    def test_batch_single(self, client):
        resp = client.post("/predict/batch", json={"texts": ["Hello world."]})
        assert resp.status_code == 200
        assert resp.json()["count"] == 1


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


class TestModelInfo:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_name" in data
        assert "backend" in data
        assert "metrics" in data
        assert data["cache_enabled"] is True


class TestFeedback:
    def test_submit_feedback(self, client):
        resp = client.post("/feedback", json={
            "text": "Test claim sentence.",
            "predicted_is_claim": True,
            "correct_is_claim": False,
            "comment": "This was an opinion, not a claim.",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"
        assert resp.json()["feedback_id"] > 0


class TestValidation:
    def test_empty_text_rejected(self, client):
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_text_rejected(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_batch_empty_list_rejected(self, client):
        resp = client.post("/predict/batch", json={"texts": []})
        assert resp.status_code == 422
