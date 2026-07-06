"""Prometheus metrics for the OCR service.

All metric objects are defined here so they are registered once at import
time and can be imported from both app.py and the pipeline modules without
circular dependency issues.

Exposed at GET /metrics in Prometheus text format.
"""
from __future__ import annotations

from prometheus_client import Counter, Histogram, make_asgi_app

# Total requests, split by document type (image/pdf) and outcome.
REQUESTS = Counter(
    "ocr_requests_total",
    "Total document processing requests",
    ["type", "outcome"],  # outcome: success | cache_hit | error
)

# End-to-end wall time for a full pipeline run (excludes cache hits).
PROCESSING_SECONDS = Histogram(
    "ocr_processing_seconds",
    "End-to-end document processing time in seconds",
    ["type"],
    buckets=[0.5, 1, 2, 5, 10, 20, 30, 60, 120],
)

# LLM fields extracted per document page.
FIELDS_EXTRACTED = Histogram(
    "ocr_fields_extracted_per_document",
    "Number of dynamic fields extracted by the LLM per document page",
    ["type"],
    buckets=[0, 1, 5, 10, 15, 20, 30, 50],
)

# OCR text regions detected per document page.
REGIONS_DETECTED = Histogram(
    "ocr_regions_per_document",
    "Number of OCR text regions detected per document page",
    ["type"],
    buckets=[5, 10, 25, 50, 75, 100, 150, 200],
)

# Successful LLM calls — tracks which model is actually doing the work
# and whether the VLM or text fallback path was used.
LLM_CALLS = Counter(
    "ocr_llm_calls_total",
    "Successful LLM inference calls by model and path",
    ["model", "path"],  # path: vlm | text
)

# ASGI sub-app that serves the /metrics endpoint in Prometheus text format.
metrics_app = make_asgi_app()
