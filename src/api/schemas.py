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
    model: str | None = Field(
        default=None,
        description="Model to use (distilbert-base-uncased, bert-base-uncased, ModernBERT-base, ensemble). Default: distilbert-base-uncased",
    )


class TokenAttribution(BaseModel):
    token: str
    weight: float


class PredictResponse(BaseModel):
    is_claim: bool = Field(description="Whether the sentence is a factual claim")
    confidence: float = Field(description="Model confidence (0.0 to 1.0)")
    model: str = Field(description="Model used for prediction")
    attribution: list[TokenAttribution] | None = Field(
        default=None, description="Top tokens that influenced the prediction"
    )


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)


class ModelPrediction(BaseModel):
    model: str
    is_claim: bool
    confidence: float


class CompareResponse(BaseModel):
    text: str
    predictions: list[ModelPrediction]
    ensemble: ModelPrediction


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of sentences to classify (max 100)",
    )
    model: str | None = Field(default=None)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int = Field(description="Number of predictions returned")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    default_model: str
    available_models: list[str]
    calibration_temperature: float
    metrics: dict | None = None


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000)
    predicted_is_claim: bool
    correct_is_claim: bool
    comment: str = Field(default="", max_length=500)


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Paragraph or article to analyze",
    )
    model: str | None = Field(default=None)


class SentenceResult(BaseModel):
    text: str
    is_claim: bool
    confidence: float


class AnalyzeResponse(BaseModel):
    sentences: list[SentenceResult]
    total: int
    claims: int
    non_claims: int


class ErrorResponse(BaseModel):
    detail: str
