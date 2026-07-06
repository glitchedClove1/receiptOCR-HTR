FROM python:3.11-slim

# System deps for OpenCV, PaddlePaddle runtime, and the health-check curl call.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install the package + heavy OCR extras.
# Copying pyproject.toml before src/ keeps this layer cached across code changes.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir ".[ocr]"

# Non-root user for runtime security.
RUN useradd --create-home --uid 10001 appuser

# Persistent directories for the result cache and generated Excel files.
# Mount both as named volumes in production so data survives container restarts.
ENV OCR_CACHE_DIR=/srv/.ocr_cache \
    OCR_OUTPUT_DIR=/srv/outputs \
    OCR_OLLAMA_BASE_URL=http://ollama:11434

RUN mkdir -p /srv/.ocr_cache /srv/outputs && chown -R appuser:appuser /srv
USER appuser

VOLUME ["/srv/.ocr_cache", "/srv/outputs"]

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# One worker = one shared in-process model cache (PaddleOCR, TrOCR).
# Scale horizontally with multiple container replicas behind a load balancer
# rather than increasing --workers, which would load the models N times.
CMD ["uvicorn", "ocr_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
