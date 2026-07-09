<div align="center">

# Financial OCR Service

**Turn scanned invoices and receipts into structured data and ready-to-use Excel workbooks — entirely on your own infrastructure.**

[![CI](https://img.shields.io/badge/CI-lint%20%7C%20typecheck%20%7C%20test-blue)](.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Proprietary-lightgrey)](pyproject.toml)

</div>

---

## Overview

This service extracts structured financial data from scanned documents — invoices, receipts, delivery notes — in both **printed and handwritten** form, across **French, English, and Arabic**. It combines classical OCR with a local vision-language model so that field extraction adapts to whatever fields actually appear on a document, instead of relying on a fixed schema.

Everything runs **locally**: no document data is ever sent to a third-party API.

- **Accepts** image uploads (JPEG/PNG/WebP/TIFF/BMP) and multi-page PDFs
- **Recognizes** printed text via PaddleOCR, with a TrOCR handwriting fallback for low-confidence Latin-script regions
- **Extracts fields dynamically** with a local vision-language model (Ollama), so the output adapts to each document's actual content rather than a fixed set of columns
- **Outputs** a per-document Excel workbook plus a structured JSON response
- **Flags** uncertain values for human review instead of guessing silently

> **Scope note:** handwritten Arabic is intentionally not handled in this version. Printed Arabic (via PaddleOCR) works; low-confidence Arabic regions are flagged for human review. See [Extending](#extending-add-arabic-or-other-handwriting-recognition) to add an Arabic HTR model.

---

## Table of contents

- [Pipeline](#pipeline)
- [Project layout](#project-layout)
- [Local setup](#local-setup)
- [Running the service](#running-the-service)
- [Processing documents](#processing-documents)
- [Docker](#docker)
- [Configuration](#configuration)
- [Observability](#observability)
- [Operations](#operations)
- [Quality tooling](#quality-tooling)
- [Extending: add Arabic (or other) handwriting recognition](#extending-add-arabic-or-other-handwriting-recognition)
- [Known limitations](#known-limitations--tuning)

---

## Pipeline

```
 image bytes ──────────────────────────────┐
 PDF bytes → rasterize each page (PyMuPDF) ─┤  (one page = one document)
                                            ▼
   preprocessing        OpenCV: grayscale, CLAHE, denoise, deskew
        │
        ▼
   PaddleOCR            text regions + confidence scores
        │
        ▼
   HTR fallback         TrOCR re-recognizes low-confidence Latin regions
        │
        ▼
   correction           numeric-context fixes, number/date/currency normalization
        │
        ▼
   rule-based extraction   spatial heuristics → vendor, dates, totals, line items
        │
        ▼
   VLM / LLM extraction    local Ollama model reads the image directly and
                           extracts every labeled field present on the document
        │
        ▼
   Excel workbook        Fields (LLM-driven) + LineItems sheets, per document
```

The LLM/VLM stage tries `qwen2.5vl:3b` against the raw document image first — this avoids compounding OCR errors, since the model sees the original pixels rather than possibly-misread text. If the vision model is unavailable, it falls back to a text-only model reading the OCR output, and finally degrades gracefully to an empty result if neither is reachable — the rest of the pipeline never blocks on it.

---

## Project layout

```
src/ocr_service/
  app.py                  FastAPI service (versioned /v1 routes, rate limiting, metrics)
  auth.py                 API-key auth dependency (stub)
  metrics.py              Prometheus metrics (requests, latency, LLM calls)
  exceptions.py           domain exception hierarchy
  logging_config.py       structured JSON logging + request-id context
  pipeline/
    config.py             pydantic settings (env-overridable)
    schemas.py             Region / Document / FieldValue / LineItem
    preprocessing.py      OpenCV steps (each toggleable)
    pdf_render.py          PDF → per-page images (PyMuPDF)
    model_loader.py        thread-safe lazy loaders (Paddle + TrOCR)
    ocr_engine.py          PaddleOCR wrapper
    script_detect.py       Latin vs Arabic heuristic
    htr_fallback.py        TrOCR fallback + script→recognizer registry
    correction.py          numeric correction + FR/EN number/date/currency
    extraction.py          spatial field & line-item extraction (rule-based)
    llm_extraction.py      VLM/text LLM dynamic field extraction (Ollama)
    excel_writer.py        openpyxl workbook (one per document)
    cache.py               SHA-256 keyed disk cache (pluggable)
    orchestrator.py        stage glue with per-stage error handling
tests/                     pytest unit, service, and (optional) end-to-end
pyproject.toml             packaging + ruff/mypy/pytest config
Dockerfile  docker-compose.yml  Makefile  .github/workflows/ci.yml
```

---

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
# Fast install for development/tests (no heavy models):
pip install -e ".[dev]"
# Full install including PaddleOCR + TrOCR runtime:
pip install -e ".[ocr,dev]"
```

(Or use the Makefile: `make dev` / `make install`.)

Model weights download automatically on first run:

- **PaddleOCR** language packs (fr/en/ar) into `~/.paddlex/official_models`.
- **TrOCR** (`microsoft/trocr-large-handwritten`) into the HuggingFace cache. On a CPU-only or low-memory machine, set `OCR_TROCR_MODEL_NAME=microsoft/trocr-base-handwritten` for speed.
- **Ollama models** (`qwen2.5vl:3b` + `qwen2.5:3b`) must be pulled separately — see [Running the service](#running-the-service).

---

## Running the service

1. **Start Ollama** and pull the models used for field extraction:

   ```bash
   ollama pull qwen2.5vl:3b   # vision-language model (primary)
   ollama pull qwen2.5:3b     # text-only fallback
   ```

2. **Start the API:**

   ```bash
   uvicorn ocr_service.app:app --host 0.0.0.0 --port 8000
   ```

   PaddleOCR is warmed up automatically at startup so the first real request isn't slowed down by a cold model load.

3. **Confirm it's healthy:**

   ```bash
   curl http://localhost:8000/health
   ```

> **Hardware note:** the VLM needs roughly 3–4 GB of VRAM. On a 4 GB GPU, stick to `3b`-class models — 7B+ models will crash Ollama with an out-of-memory error. CPU-only inference works but is significantly slower (tens of seconds to minutes per document).

---

## Processing documents

### A single image

```bash
curl -s -F "file=@samples/sample_receipt.jpg" http://localhost:8000/v1/process | jq
```

The response includes the structured `document` (with dynamically extracted `extra_fields`) and an `excel_filename`. Every response carries an `X-Request-ID` header for log correlation.

Download the workbook:

```bash
curl -OJ http://localhost:8000/v1/download/<image_hash>.xlsx
```

Re-uploading the same image returns `"cache_hit": true` and reuses the existing workbook.

If `OCR_API_KEY` is set, pass it on every `/v1` route:

```bash
curl -s -H "X-API-Key: <key>" -F "file=@receipt.jpg" http://localhost:8000/v1/process | jq
```

### A PDF

Each page comes back as its own entry with its own workbook:

```bash
curl -s -F "file=@invoice.pdf" http://localhost:8000/v1/process-pdf | jq
```

The response has `page_count` and a `pages` array; each element has a `page_number`, the extracted `document`, and an `excel_filename` of the form `<pdf_hash>_p<N>.xlsx`. Download any page's workbook via the same download endpoint:

```bash
curl -OJ http://localhost:8000/v1/download/<pdf_hash>_p1.xlsx
```

Re-uploading the same PDF returns `"cache_hit": true` and reuses every page's workbook. `OCR_PDF_RENDER_DPI` (default `220`) and `OCR_PDF_MAX_PAGES` (default `50`) control rasterization sharpness and the per-file page cap.

### Output fields

The **Fields** sheet of each workbook is driven entirely by what the LLM/VLM extracted from the document — there is no fixed template of rows. Values the model flagged as uncertain are highlighted for human review.

Generated workbooks and the result cache live under `./.ocr_cache` (configurable via `OCR_CACHE_DIR`).

---

## Docker

```bash
docker compose up --build
```

The compose stack brings up three services:

| Service | Purpose |
|---|---|
| `ollama` | Serves the VLM/LLM used for field extraction |
| `ollama-init` | One-shot init container that pulls the required models on first boot |
| `ocr` | The FastAPI service itself |

Named volumes persist the Ollama models, the result cache, and the PaddleOCR/HuggingFace model weights across restarts, so nothing re-downloads after the first `docker compose up`.

---

## Configuration

All settings are environment variables prefixed `OCR_` (or a `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `OCR_CONFIDENCE_THRESHOLD` | `0.80` | Below this Paddle confidence, Latin regions go to TrOCR |
| `OCR_PADDLE_PRIMARY_LANG` | `fr` | PaddleOCR primary language pack |
| `OCR_TROCR_MODEL_NAME` | `microsoft/trocr-large-handwritten` | HTR model |
| `OCR_PRE_GRAYSCALE` / `_CLAHE` / `_DENOISE` / `_DESKEW` | `true` | Toggle each preprocessing step |
| `OCR_CACHE_DIR` | `./.ocr_cache` | Cache + workbook directory |
| `OCR_CACHE_TTL_SECONDS` | `604800` | Cache lifetime; `0` = never expire |
| `OCR_USE_GPU` | `false` | Use GPU for Paddle + TrOCR |
| `OCR_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `OCR_PDF_RENDER_DPI` | `220` | DPI for rasterizing PDF pages before OCR |
| `OCR_PDF_MAX_PAGES` | `50` | Max pages processed from one PDF |
| `OCR_LOG_LEVEL` | `INFO` | Log level for the JSON logger |
| `OCR_API_KEY` | _(empty)_ | If set, `/v1` routes require `X-API-Key`; empty disables auth |
| `OCR_OLLAMA_ENABLED` | `true` | Set `false` to skip LLM/VLM field extraction entirely |
| `OCR_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OCR_VLM_MODEL` | `qwen2.5vl:3b` | Vision-language model for primary extraction; set `""` to use text-only |
| `OCR_OLLAMA_MODEL` | `qwen2.5:3b` | Text-only fallback model |
| `OCR_OLLAMA_TIMEOUT_SECONDS` | `180` | Timeout for each Ollama call |

> Config is read once via `lru_cache` — **restart the service** after changing environment variables or `.env`.

---

## Observability

- **Metrics** — Prometheus-format metrics are exposed at `GET /metrics`: request counts by outcome, end-to-end processing latency, fields/regions extracted per document, and which model (VLM vs. text fallback) served each request.
- **Rate limiting** — `/v1/process` and `/v1/process-pdf` are rate-limited per client; exceeding the limit returns `429`.
- **Logging** — structured JSON on stdout, one object per line, each tagged with the request's `request_id` (also returned as the `X-Request-ID` response header). Pipe into any log aggregator.

---

## Operations

- **Errors** map to distinct statuses: `400` invalid/empty/malformed upload, `413` too large, `415` unsupported type, `401` bad API key, `429` rate limited, `503` model failed to load, `500` other pipeline failure. Clients never receive a stack trace.
- **Concurrency:** models are loaded once behind a lock and shared in-process; PaddleOCR inference is serialized internally since the underlying predictor is not thread-safe. Processing runs in a background thread pool so the event loop stays responsive under concurrent requests. Run one uvicorn worker per container and scale with replicas behind a load balancer, so each process keeps a single warm model cache.
- **Startup warmup:** PaddleOCR is loaded during application startup rather than on the first request, so cold-start latency doesn't land on your first real user.

---

## Quality tooling

```bash
make lint        # ruff
make typecheck   # mypy (advisory)
make test        # pytest
```

CI (`.github/workflows/ci.yml`) runs lint + type check + tests on every push and pull request, installing only the lightweight `[dev]` extra.

## Tests

```bash
pytest -q
```

The pure-logic tests run without the heavy OCR dependencies. The end-to-end test activates only if `samples/sample_receipt.jpg` exists and Paddle/TrOCR are installed.

---

## Extending: add Arabic (or other) handwriting recognition

The fallback router in `htr_fallback.py` maps a `Script` to a recognizer callable `(crop: np.ndarray) -> str`. To add handwritten Arabic:

1. Implement a recognizer (load your Arabic HTR model, return text for a crop).
2. Register it in `build_registry`:
   ```python
   registry[Script.ARABIC] = my_arabic_recognizer
   ```

No other code changes are needed — low-confidence Arabic regions will then be routed instead of flagged.

---

## Known limitations / tuning

- **Spatial extraction is heuristic.** The rule-based line-item and total rules are tuned for common FR/EN receipt and invoice layouts; expect to adjust the keyword cues in `extraction.py` and the row-grouping tolerance for your real documents. The LLM/VLM extraction stage is more robust to layout variation but is not infallible either.
- Confusable-character correction only fires inside digit-dominant tokens, so it won't fix a misread letter inside a word.
- Day-first is assumed for ambiguous numeric dates (FR/EU convention).
- The VLM path requires a local Ollama installation with sufficient VRAM (or patience, on CPU); see the hardware note under [Running the service](#running-the-service).
