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

# Result cache directory (documents + workbooks, served via /v1/download).
ENV OCR_CACHE_DIR=/srv/.ocr_cache \
    OCR_OLLAMA_BASE_URL=http://ollama:11434

# Pre-create every directory a named volume mounts over, owned by appuser —
# Docker creates missing mountpoints as root, which would break model-weight
# downloads for the non-root user. PaddleOCR 3.x caches under ~/.paddlex.
RUN mkdir -p /srv/.ocr_cache \
        /home/appuser/.cache/huggingface \
        /home/appuser/.paddlex \
    && chown -R appuser:appuser /srv /home/appuser
USER appuser

VOLUME ["/srv/.ocr_cache"]

EXPOSE 8000

# start-period covers first-boot PaddleOCR weight downloads (UVDoc,
# PP-OCRv6 det/rec, textline orientation) from an empty paddle_cache volume,
# which took ~60s in testing — 30s was too tight and caused a transient
# "unhealthy" flap before the app finished starting.
HEALTHCHECK --interval=20s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# One worker = one shared in-process model cache (PaddleOCR, TrOCR).
# Scale horizontally with multiple container replicas behind a load balancer
# rather than increasing --workers, which would load the models N times.
CMD ["uvicorn", "ocr_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
