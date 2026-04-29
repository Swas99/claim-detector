# Multi-stage Dockerfile for claim-detector API
#
# Prerequisites: train at least DistilBERT before building:
#   make train-distilbert
#
# Build: docker build -t claim-detector .
# Run:   docker run -p 8000:8000 claim-detector

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/

# Copy model weights (must exist locally — run make train-distilbert first)
COPY models/distilbert-base-uncased/ models/distilbert-base-uncased/
COPY models/calibration.json models/calibration.json

# Create directories for optional models
RUN mkdir -p data

# Environment
ENV PYTHONUNBUFFERED=1
ENV CLAIM_ACTIVE_MODEL=distilbert-base-uncased

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
