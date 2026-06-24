# Financial OCR Service

OCR pipeline for financial/accounting documents and receipts containing both
**printed text and handwriting**. PaddleOCR runs first; low-confidence
**Latin-script** regions are re-recognized with **TrOCR**. Output is structured
fields plus a per-document Excel workbook. Everything runs **locally** — no
data leaves the host.

> **Scope note:** handwritten Arabic is intentionally **not** handled in this
> version. Printed Arabic (via PaddleOCR) works; low-confidence Arabic regions
> are flagged `low_confidence_arabic` for human review. The recognizer router
> has a registration slot to add an Arabic HTR model later (see *Extending*).

## Pipeline

```
image bytes
  → preprocessing (OpenCV: grayscale, CLAHE, denoise, deskew)
  → PaddleOCR (regions + confidence)
  → HTR fallback (TrOCR on low-confidence Latin regions only)
  → correction (numeric-context fixes, number/date/currency normalization)
  → extraction (rule-based + spatial heuristics → fields, line items, totals)
  → Excel workbook (Fields + LineItems sheets)
```

## Project layout

```
src/ocr_service/
  app.py                  FastAPI service (versioned /v1 routes)
  auth.py                 API-key auth dependency (stub)
  exceptions.py           domain exception hierarchy
  logging_config.py       structured JSON logging + request-id context
  pipeline/
    config.py             pydantic settings (env-overridable)
    schemas.py            Region / Document / FieldValue / LineItem
    preprocessing.py      OpenCV steps (each toggleable)
    model_loader.py       thread-safe lazy loaders (Paddle + TrOCR)
    ocr_engine.py         PaddleOCR wrapper
    script_detect.py      Latin vs Arabic heuristic
    htr_fallback.py       TrOCR fallback + script→recognizer registry
    correction.py         numeric correction + FR/EN number/date/currency
    extraction.py         spatial field & line-item extraction
    excel_writer.py       openpyxl workbook (one per document)
    cache.py              SHA-256 keyed disk cache (pluggable)
    orchestrator.py       stage glue with per-stage error handling
tests/                    pytest unit, service, and (optional) end-to-end
pyproject.toml            packaging + ruff/mypy/pytest config
Dockerfile  docker-compose.yml  Makefile  .github/workflows/ci.yml
```

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
- **PaddleOCR** language packs (fr/en/ar) into `~/.paddleocr`.
- **TrOCR** (`microsoft/trocr-large-handwritten`) into the HuggingFace cache.
  On a CPU-only or low-memory machine, set
  `OCR_TROCR_MODEL_NAME=microsoft/trocr-base-handwritten` for speed.

## Run

```bash
uvicorn ocr_service.app:app --host 0.0.0.0 --port 8000
```

Process a document:

```bash
curl -s -F "file=@samples/sample_receipt.jpg" http://localhost:8000/v1/process | jq
```

The response includes the structured `document` and an `excel_filename`, and
every response carries an `X-Request-ID` header for log correlation.
Download the workbook:

```bash
curl -OJ http://localhost:8000/v1/download/<image_hash>.xlsx
```

Re-uploading the same image returns `"cache_hit": true` and reuses the
existing workbook.

If `OCR_API_KEY` is set, pass it on `/v1` routes:

```bash
curl -s -H "X-API-Key: <key>" -F "file=@receipt.jpg" \
  http://localhost:8000/v1/process | jq
```


## Docker

```bash
docker compose up --build
```

Volumes persist the result cache and the downloaded model weights across
restarts.

## Configuration

All settings are environment variables prefixed `OCR_` (or a `.env` file):

| Variable | Default | Meaning |
|---|---|---|
| `OCR_CONFIDENCE_THRESHOLD` | `0.80` | Below this Paddle confidence, Latin regions go to TrOCR |
| `OCR_PADDLE_PRIMARY_LANG` | `fr` | PaddleOCR primary language pack |
| `OCR_TROCR_MODEL_NAME` | `microsoft/trocr-large-handwritten` | HTR model |
| `OCR_PRE_GRAYSCALE/CLAHE/DENOISE/DESKEW` | `true` | Toggle each preprocessing step |
| `OCR_CACHE_DIR` | `./.ocr_cache` | Cache + workbook directory |
| `OCR_CACHE_TTL_SECONDS` | `604800` | Cache lifetime; `0` = never expire |
| `OCR_USE_GPU` | `false` | Use GPU for Paddle + TrOCR |
| `OCR_MAX_UPLOAD_MB` | `25` | Upload size limit |
| `OCR_LOG_LEVEL` | `INFO` | Log level for the JSON logger |
| `OCR_API_KEY` | _(empty)_ | If set, `/v1` routes require `X-API-Key`; empty disables auth |

## Operations

- **Logging** is structured JSON on stdout, one object per line, each tagged
  with the request's `request_id` (also returned as the `X-Request-ID`
  response header). Pipe into any log aggregator.
- **Errors** map to distinct statuses: 400 invalid/empty image, 413 too large,
  415 unsupported type, 401 bad API key, 503 model failed to load, 500 other
  pipeline failure. Clients never receive a stack trace.
- **Concurrency:** models are loaded once behind a lock and shared in-process.
  Run one uvicorn worker per container and scale with replicas behind a load
  balancer, so each process keeps a single warm model cache.

## Quality tooling

```bash
make lint        # ruff
make typecheck   # mypy (advisory)
make test        # pytest
```

CI (`.github/workflows/ci.yml`) runs lint + type check + tests on every push
and pull request, installing only the lightweight `[dev]` extra.

## Tests

```bash
pytest -q
```

The pure-logic tests run without the heavy OCR dependencies. The end-to-end
test activates only if `samples/sample_receipt.jpg` exists and Paddle/TrOCR
are installed.

## Extending: add Arabic (or other) handwriting recognition

The fallback router in `htr_fallback.py` maps a `Script` to a recognizer
callable `(crop: np.ndarray) -> str`. To add handwritten Arabic:

1. Implement a recognizer (load your Arabic HTR model, return text for a crop).
2. Register it in `build_registry`:
   ```python
   registry[Script.ARABIC] = my_arabic_recognizer
   ```

No other code changes are needed — low-confidence Arabic regions will then be
routed instead of flagged.

## Known limitations / tuning

- **Spatial extraction is heuristic.** The line-item and total rules are tuned
  for common FR/EN receipt and invoice layouts; expect to adjust the keyword
  cues in `extraction.py` and the row-grouping tolerance `y_tol` for your real
  documents.
- Confusable-character correction only fires inside digit-dominant tokens, so
  it won't fix a misread letter inside a word.
- Day-first is assumed for ambiguous numeric dates (FR/EU convention).
