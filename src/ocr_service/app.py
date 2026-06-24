"""FastAPI service.

Wires the pipeline behind a versioned HTTP API with:
  * structured JSON logging + a per-request correlation id (X-Request-ID),
  * granular exception handling mapping domain errors to HTTP statuses,
  * an optional API-key auth dependency (stub),
  * SHA-256 content-addressed caching of results and workbooks.

Routes are under /v1 so the response shape can evolve without breaking
existing callers.
"""
from __future__ import annotations

import os
import uuid

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import __version__
from .auth import require_api_key
from .exceptions import (
    InvalidImageError,
    ModelLoadError,
    PayloadTooLargeError,
    PipelineError,
    UnsupportedMediaError,
)
from .logging_config import configure_logging, get_logger, request_id_var
from .pipeline.cache import get_cache, image_hash
from .pipeline.config import get_settings
from .pipeline.excel_writer import write_workbook
from .pipeline.orchestrator import process_image_bytes
from .pipeline.schemas import ProcessResponse

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

app = FastAPI(title="Financial OCR Service", version=__version__)
cache = get_cache(settings)
router = APIRouter(prefix="/v1")

_ALLOWED = {"image/jpeg", "image/png", "image/jpg", "image/webp",
            "image/tiff", "image/bmp"}


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Assign/propagate a correlation id and echo it back in the response."""
    rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    token = request_id_var.set(rid)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = rid
    return response


# --- Exception handlers: domain errors -> HTTP statuses --------------------
@app.exception_handler(UnsupportedMediaError)
async def _unsupported(_: Request, exc: UnsupportedMediaError):
    return JSONResponse(status_code=415, content={"detail": str(exc)})


@app.exception_handler(PayloadTooLargeError)
async def _too_large(_: Request, exc: PayloadTooLargeError):
    return JSONResponse(status_code=413, content={"detail": str(exc)})


@app.exception_handler(InvalidImageError)
async def _bad_image(_: Request, exc: InvalidImageError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelLoadError)
async def _model_load(_: Request, exc: ModelLoadError):
    log.error("model load failed", fields={"error": str(exc)})
    return JSONResponse(status_code=503, content={"detail": "Model unavailable."})


@app.exception_handler(PipelineError)
async def _pipeline(_: Request, exc: PipelineError):
    log.error("pipeline error", fields={"error": str(exc)})
    return JSONResponse(status_code=500, content={"detail": "Processing failed."})


# --- Routes ----------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.post("/process", response_model=ProcessResponse,
             dependencies=[Depends(require_api_key)])
async def process(file: UploadFile = File(...)) -> ProcessResponse:
    if file.content_type not in _ALLOWED:
        raise UnsupportedMediaError(f"Unsupported content type: {file.content_type}")

    data = await file.read()
    if not data:
        raise InvalidImageError("Empty upload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise PayloadTooLargeError("File too large.")

    key = image_hash(data)
    excel_name = f"{key}.xlsx"

    cached = cache.get(key)
    if cached is not None and cache.has_excel(key):
        log.info("cache hit", fields={"image_hash": key})
        return ProcessResponse(cache_hit=True, image_hash=key,
                               excel_filename=excel_name, document=cached)

    doc = process_image_bytes(data, settings)  # raises domain errors
    write_workbook(doc, cache.excel_path(key), settings)
    cache.put(key, doc)
    log.info("processed", fields={"image_hash": key, "regions": doc.region_count,
                                  "low_confidence": doc.low_confidence_count})

    return ProcessResponse(cache_hit=False, image_hash=key,
                           excel_filename=excel_name, document=doc)


@router.get("/download/{name}", dependencies=[Depends(require_api_key)])
def download(name: str) -> FileResponse:
    # Prevent path traversal: only bare '<hash>.xlsx' names are allowed.
    if os.path.basename(name) != name or not name.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    path = os.path.join(settings.cache_dir, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found.")
    return FileResponse(
        path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=name,
    )


app.include_router(router)
