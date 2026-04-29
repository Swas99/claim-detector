"""API route definitions."""

import json
import re

from fastapi import APIRouter, Request

from src.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    CompareRequest,
    CompareResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    ModelPrediction,
    PredictRequest,
    PredictResponse,
    SentenceResult,
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

    if body.model == "ensemble":
        result = detector.predict_ensemble(body.text)
    else:
        result = detector.predict(body.text, model_name=body.model)

    return PredictResponse(
        is_claim=result.is_claim,
        confidence=round(result.confidence, 4),
        model=result.model,
        attribution=result.attribution,
    )


@router.post("/compare", response_model=CompareResponse)
async def compare(body: CompareRequest, request: Request):
    """Run all models on a sentence and return side-by-side comparison."""
    detector = get_detector(request)
    result = detector.compare(body.text)

    predictions = [
        ModelPrediction(
            model=name,
            is_claim=pred.is_claim,
            confidence=round(pred.confidence, 4),
        )
        for name, pred in result.predictions.items()
    ]

    return CompareResponse(
        text=result.text,
        predictions=predictions,
        ensemble=ModelPrediction(
            model="ensemble",
            is_claim=result.ensemble.is_claim,
            confidence=round(result.ensemble.confidence, 4),
        ),
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(body: BatchPredictRequest, request: Request):
    """Classify multiple sentences in a single request (max 100)."""
    detector = get_detector(request)
    results = [
        detector.predict_ensemble(t) if body.model == "ensemble"
        else detector.predict(t, model_name=body.model)
        for t in body.texts
    ]
    predictions = [
        PredictResponse(
            is_claim=r.is_claim,
            confidence=round(r.confidence, 4),
            model=r.model,
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
        model_loaded=len(detector._models) > 0,
    )


@router.get("/model/info", response_model=ModelInfoResponse)
async def model_info(request: Request):
    """Return model metadata and evaluation metrics."""
    detector = get_detector(request)
    info = detector.info

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
        default_model=info["default_model"],
        available_models=info["available_models"],
        calibration_temperature=info["calibration_temperature"],
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


def _split_sentences(text: str) -> list[str]:
    """Split a paragraph into sentences. Simple regex-based splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_paragraph(body: AnalyzeRequest, request: Request):
    """Analyze a paragraph: split into sentences and classify each one."""
    detector = get_detector(request)
    sentences = _split_sentences(body.text)

    results = []
    for sent in sentences:
        if body.model == "ensemble":
            pred = detector.predict_ensemble(sent)
        else:
            pred = detector.predict(sent, model_name=body.model)
        results.append(SentenceResult(
            text=sent,
            is_claim=pred.is_claim,
            confidence=round(pred.confidence, 4),
        ))

    claims = sum(1 for r in results if r.is_claim)
    return AnalyzeResponse(
        sentences=results,
        total=len(results),
        claims=claims,
        non_claims=len(results) - claims,
    )
