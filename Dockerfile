FROM python:3.11-slim AS base

# System deps for OpenCV and Paddle runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
# Install the package plus the heavy OCR extras.
RUN pip install --no-cache-dir ".[ocr]"

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser
ENV OCR_CACHE_DIR=/srv/.ocr_cache
RUN mkdir -p /srv/.ocr_cache && chown -R appuser:appuser /srv
USER appuser

# Model weights download on first request; mount a volume here to persist
# them and the cache across container restarts.
VOLUME ["/srv/.ocr_cache"]

EXPOSE 8000
# A single worker keeps one shared in-process model cache. Scale out with
# replicas behind a load balancer rather than many workers per container.
CMD ["uvicorn", "ocr_service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
