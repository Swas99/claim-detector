"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Natural language sentence to classify",
        examples=["The Empire State Building is the tallest building in New York City."],
    )


class PredictResponse(BaseModel):
    is_claim: bool = Field(description="Whether the sentence is a factual claim")
    confidence: float = Field(description="Model confidence (0.0 to 1.0)")
    source: str = Field(description="Prediction source: model, fast_filter, or cache")
    cached: bool = Field(default=False, description="Whether result came from cache")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of sentences to classify (max 100)",
    )


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int = Field(description="Number of predictions returned")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    backend: str
    max_length: int
    calibration_temperature: float
    cache_enabled: bool
    fast_filter_enabled: bool
    cache_stats: dict | None = None
    metrics: dict | None = None


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    predicted_is_claim: bool
    correct_is_claim: bool
    comment: str = Field(default="", max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class ErrorResponse(BaseModel):
    detail: str
