# Dockerfile for claim-detector API
#
# Prerequisites: train at least DistilBERT before building:
#   make train-distilbert
#
# Build: docker build -t claim-detector .
# Run:   docker run -p 8000:8000 claim-detector

FROM python:3.13-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy application code
COPY src/ src/

# Copy model weights (must exist locally)
COPY models/distilbert-base-uncased/ models/distilbert-base-uncased/
COPY models/calibration.json models/calibration.json

RUN mkdir -p data

ENV PYTHONUNBUFFERED=1
ENV CLAIM_ACTIVE_MODEL=distilbert-base-uncased

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
