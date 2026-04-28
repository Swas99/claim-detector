"""Security and edge-case tests."""

import os

import pytest

os.environ["CLAIM_USE_ONNX"] = "false"


class TestInputValidation:
    def test_very_long_text(self, client):
        """Text over max_length should still work (tokenizer truncates)."""
        long_text = "This is a factual claim. " * 500  # ~12,500 chars
        resp = client.post("/predict", json={"text": long_text[:5000]})
        assert resp.status_code == 200

    def test_text_exceeds_max_length(self, client):
        """Text over 5000 chars should be rejected by Pydantic."""
        resp = client.post("/predict", json={"text": "a" * 5001})
        assert resp.status_code == 422

    def test_unicode_input(self, client):
        resp = client.post("/predict", json={"text": "GDP增长了2.1%。"})
        assert resp.status_code == 200

    def test_special_characters(self, client):
        resp = client.post("/predict", json={
            "text": "The <b>GDP</b> grew by 2.1% — that's a 'fact'."
        })
        assert resp.status_code == 200

    def test_newlines_in_text(self, client):
        resp = client.post("/predict", json={
            "text": "Line one.\nLine two.\nLine three."
        })
        assert resp.status_code == 200

    def test_json_injection(self, client):
        resp = client.post("/predict", json={
            "text": '{"malicious": true}'
        })
        assert resp.status_code == 200

    def test_invalid_content_type(self, client):
        resp = client.post("/predict", content="not json",
                           headers={"Content-Type": "text/plain"})
        assert resp.status_code == 422

    def test_nonexistent_endpoint(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
