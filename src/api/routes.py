"""API route definitions."""

import json
import time

from fastapi import APIRouter, Request

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from src.config import settings

router = APIRouter()


def get_detector(request: Request):
    return request.app.state.detector


def get_feedback_store(request: Request):
    return request.app.state.feedback_store


@router.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest, request: Request):
    """Classify a single sentence as claim or not."""
    detector = get_detector(request)
    result = detector.predict(body.text)
    return PredictResponse(
        is_claim=result.is_claim,
        confidence=round(result.confidence, 4),
        source=result.source,
        cached=result.cached,
        attribution=result.attribution,
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(body: BatchPredictRequest, request: Request):
    """Classify multiple sentences in a single request (max 100)."""
    detector = get_detector(request)
    results = detector.predict_batch(body.texts)
    predictions = [
        PredictResponse(
            is_claim=r.is_claim,
            confidence=round(r.confidence, 4),
            source=r.source,
            cached=r.cached,
        )
        for r in results
    ]
    return BatchPredictResponse(predictions=predictions, count=len(predictions))


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Liveness and readiness check."""
    detector = get_detector(request)
    return HealthResponse(
        status="healthy",
        model_loaded=detector.runner.model is not None,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    """Return model metadata, backend, and cache stats."""
    detector = get_detector(request)
    info = detector.info

    # Load evaluation metrics if available
    metrics_path = settings.model_dir / settings.active_model / "metrics.json"
    metrics = None
    if metrics_path.exists():
        try:
            with open(metrics_path) as f:
                raw = json.load(f)
            metrics = {
                "accuracy": raw.get("accuracy"),
                "precision": raw.get("precision"),
                "recall": raw.get("recall"),
                "f1": raw.get("f1"),
            }
        except (json.JSONDecodeError, OSError):
            metrics = None

    return ModelInfoResponse(
        model_name=info["model_name"],
        backend=info["backend"],
        max_length=info["max_length"],
        calibration_temperature=info["calibration_temperature"],
        cache_enabled=info["cache_enabled"],
        fast_filter_enabled=info["fast_filter_enabled"],
        cache_stats=info.get("cache_stats"),
        metrics=metrics,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest, request: Request):
    """Submit correction for a prediction (for model improvement)."""
    store = get_feedback_store(request)
    feedback_id = store.add(
        text=body.text,
        predicted=body.predicted_is_claim,
        correct=body.correct_is_claim,
        comment=body.comment,
    )
    return FeedbackResponse(status="recorded", feedback_id=feedback_id)
