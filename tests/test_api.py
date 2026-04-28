"""Integration tests for the API endpoints."""

import os

import pytest

os.environ["CLAIM_USE_ONNX"] = "false"


class TestPredict:
    def test_claim_sentence(self, client):
        resp = client.post("/predict", json={
            "text": "The unemployment rate fell to 3.5% in December 2023."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_claim"] is True
        assert data["confidence"] > 0.5
        assert "model" in data

    def test_non_claim_opinion(self, client):
        resp = client.post("/predict", json={
            "text": "I think chocolate ice cream is the best."
        })
        assert resp.status_code == 200
        assert resp.json()["is_claim"] is False

    def test_attribution_present(self, client):
        resp = client.post("/predict", json={
            "text": "GDP grew by 3.2% in the last fiscal year."
        })
        data = resp.json()
        assert data["attribution"] is not None
        assert len(data["attribution"]) > 0
        assert "token" in data["attribution"][0]
        assert "weight" in data["attribution"][0]

    def test_model_selection(self, client):
        resp = client.post("/predict", json={
            "text": "Water boils at 100 degrees Celsius.",
            "model": "distilbert-base-uncased",
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "distilbert-base-uncased"

    def test_ensemble(self, client):
        resp = client.post("/predict", json={
            "text": "The population of Tokyo is over 13 million.",
            "model": "ensemble",
        })
        assert resp.status_code == 200
        assert resp.json()["model"] == "ensemble"

    def test_response_schema(self, client):
        resp = client.post("/predict", json={"text": "Test sentence."})
        data = resp.json()
        assert "is_claim" in data
        assert "confidence" in data
        assert "model" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0


class TestCompare:
    def test_compare_returns_all_models(self, client):
        resp = client.post("/compare", json={
            "text": "The earth is round."
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "predictions" in data
        assert "ensemble" in data
        assert len(data["predictions"]) >= 1
        assert data["ensemble"]["model"] == "ensemble"


class TestBatchPredict:
    def test_batch_multiple(self, client):
        resp = client.post("/predict/batch", json={
            "texts": [
                "The earth is round.",
                "I love pizza.",
                "GDP grew 2.1%.",
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert len(data["predictions"]) == 3


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
        assert "default_model" in data
        assert "available_models" in data
        assert "ensemble" in data["available_models"]


class TestFeedback:
    def test_submit_feedback(self, client):
        resp = client.post("/feedback", json={
            "text": "Test claim sentence.",
            "predicted_is_claim": True,
            "correct_is_claim": False,
            "comment": "This was an opinion.",
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "recorded"


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
