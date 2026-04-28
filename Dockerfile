# Multi-stage Dockerfile for claim-detector API
# Stage 1: Build dependencies
# Stage 2: Slim runtime with model + API

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS builder

WORKDIR /app

# Install build deps
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ─── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.13-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY models/distilbert-base-uncased/ models/distilbert-base-uncased/
COPY models/calibration.json models/calibration.json

# Create data directory (dataset downloaded at train time, not needed for serving)
RUN mkdir -p data models/onnx

# Environment
ENV PYTHONUNBUFFERED=1
ENV CLAIM_USE_ONNX=false
ENV CLAIM_ACTIVE_MODEL=distilbert-base-uncased

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
