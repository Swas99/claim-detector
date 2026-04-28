"""FastAPI application for claim detection.

Loads the ClaimDetector on startup and serves prediction endpoints.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.routes import router
from src.config import settings
from src.engine.detector import ClaimDetector
from src.feedback.store import FeedbackStore

logger = structlog.get_logger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("loading_model", model=settings.active_model, use_onnx=settings.use_onnx)
    start = time.time()
    app.state.detector = ClaimDetector(use_onnx=settings.use_onnx)
    app.state.feedback_store = FeedbackStore()
    elapsed = time.time() - start
    logger.info("model_loaded", elapsed_s=round(elapsed, 2))
    yield
    logger.info("shutting_down")


app = FastAPI(
    title="Claim Detector API",
    description="Sentence-level claim detection: identifies whether a sentence contains a verifiable factual claim.",
    version="0.1.0",
    lifespan=lifespan,
)

# Middleware
app.state.limiter = limiter
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round(elapsed * 1000, 1),
    )
    return response


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."},
    )


# Mount routes
app.include_router(router)

# Serve demo UI
STATIC_DIR = Path(__file__).parent.parent / "static"


@app.get("/", include_in_schema=False)
async def demo_ui():
    return FileResponse(STATIC_DIR / "index.html")
