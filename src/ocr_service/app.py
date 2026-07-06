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

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import __version__
from .auth import require_api_key
from .exceptions import (
    InvalidImageError,
    InvalidPdfError,
    ModelLoadError,
    PayloadTooLargeError,
    PipelineError,
    UnsupportedMediaError,
)
from .logging_config import configure_logging, get_logger, request_id_var
from .metrics import (
    FIELDS_EXTRACTED,
    PROCESSING_SECONDS,
    REGIONS_DETECTED,
    REQUESTS,
    metrics_app,
)
from .pipeline.cache import content_hash, get_cache, image_hash, page_key
from .pipeline.config import get_settings
from .pipeline.excel_writer import write_workbook
from .pipeline.model_loader import get_paddle
from .pipeline.orchestrator import process_image_bytes, process_pdf_bytes
from .pipeline.schemas import (
    Document,
    PageResult,
    PdfProcessResponse,
    ProcessResponse,
)

settings = get_settings()
configure_logging(settings.log_level)
log = get_logger(__name__)

# Rate limiter — keyed by client IP.
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up PaddleOCR before the first request so it doesn't slow it down."""
    try:
        get_paddle(settings)
        log.info("model_warmup_complete")
    except Exception as exc:  # noqa: BLE001
        log.warning("model_warmup_failed", fields={"error": str(exc)})
    yield


app = FastAPI(title="Financial OCR Service", version=__version__, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.mount("/metrics", metrics_app)
cache = get_cache(settings)
router = APIRouter(prefix="/v1")

_ALLOWED = {"image/jpeg", "image/png", "image/jpg", "image/webp",
            "image/tiff", "image/bmp"}
_PDF_TYPES = {"application/pdf"}

# Magic byte signatures — checked against raw file bytes to prevent spoofed MIME types.
_MAGIC: list[tuple[bytes, set[str]]] = [
    (b"\xff\xd8\xff",       {"image/jpeg", "image/jpg"}),
    (b"\x89PNG\r\n\x1a\n", {"image/png"}),
    (b"%PDF",               {"application/pdf"}),
    (b"II*\x00",            {"image/tiff"}),
    (b"MM\x00*",            {"image/tiff"}),
    (b"BM",                 {"image/bmp"}),
]


def _magic_ok(data: bytes, content_type: str) -> bool:
    """Return True if the file's magic bytes match the declared content type."""
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return content_type == "image/webp"
    for magic, types in _MAGIC:
        if data.startswith(magic):
            return content_type in types
    return False


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


@app.exception_handler(InvalidPdfError)
async def _bad_pdf(_: Request, exc: InvalidPdfError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


def _doc_type(request: Request) -> str:
    return "pdf" if "pdf" in request.url.path else "image"


@app.exception_handler(ModelLoadError)
async def _model_load(request: Request, exc: ModelLoadError):
    log.error("model load failed", fields={"error": str(exc)})
    REQUESTS.labels(type=_doc_type(request), outcome="error").inc()
    return JSONResponse(status_code=503, content={"detail": "Model unavailable."})


@app.exception_handler(PipelineError)
async def _pipeline(request: Request, exc: PipelineError):
    log.error("pipeline error", fields={"error": str(exc)})
    REQUESTS.labels(type=_doc_type(request), outcome="error").inc()
    return JSONResponse(status_code=500, content={"detail": "Processing failed."})


# --- Routes ----------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.post("/process", response_model=ProcessResponse,
             dependencies=[Depends(require_api_key)])
@limiter.limit("20/minute")
async def process(request: Request, file: UploadFile = File(...)) -> ProcessResponse:
    if file.content_type not in _ALLOWED:
        raise UnsupportedMediaError(f"Unsupported content type: {file.content_type}")

    data = await file.read()
    if not data:
        raise InvalidImageError("Empty upload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise PayloadTooLargeError("File too large.")
    if not _magic_ok(data, file.content_type or ""):
        raise InvalidImageError("File content does not match declared type.")

    key = image_hash(data)
    excel_name = f"{key}.xlsx"

    cached = cache.get(key)
    if cached is not None and cache.has_excel(key):
        log.info("cache hit", fields={"image_hash": key})
        REQUESTS.labels(type="image", outcome="cache_hit").inc()
        return ProcessResponse(cache_hit=True, image_hash=key,
                               excel_filename=excel_name, document=cached)

    def _process_and_store() -> Document:
        # Everything blocking — inference, workbook serialization, cache
        # write — stays inside the executor thread so the event loop is
        # never stalled.
        d = process_image_bytes(data, settings)
        write_workbook(d, cache.excel_path(key), settings)
        cache.put(key, d)
        return d

    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    doc = await loop.run_in_executor(None, _process_and_store)
    PROCESSING_SECONDS.labels(type="image").observe(time.perf_counter() - t0)
    REQUESTS.labels(type="image", outcome="success").inc()
    REGIONS_DETECTED.labels(type="image").observe(doc.region_count)
    FIELDS_EXTRACTED.labels(type="image").observe(len(doc.extra_fields))
    log.info("processed", fields={"image_hash": key, "regions": doc.region_count,
                                  "low_confidence": doc.low_confidence_count})

    return ProcessResponse(cache_hit=False, image_hash=key,
                           excel_filename=excel_name, document=doc)


@router.post("/process-pdf", response_model=PdfProcessResponse,
             dependencies=[Depends(require_api_key)])
@limiter.limit("5/minute")
async def process_pdf(
    request: Request, file: UploadFile = File(...)
) -> PdfProcessResponse:
    """Process a PDF: each page becomes its own document + workbook.

    The whole-PDF hash gates caching; a re-uploaded identical PDF returns the
    prior per-page results and reuses the existing workbooks.
    """
    if file.content_type not in _PDF_TYPES:
        raise UnsupportedMediaError(f"Expected a PDF, got: {file.content_type}")

    data = await file.read()
    if not data:
        raise InvalidPdfError("Empty upload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise PayloadTooLargeError("File too large.")
    if not _magic_ok(data, file.content_type or ""):
        raise InvalidPdfError("File content does not match declared type.")

    pdf_hash = content_hash(data)
    loop = asyncio.get_running_loop()

    # --- Cache check ---------------------------------------------------------
    # The stored page count bounds the scan; without it a fully cached PDF is
    # indistinguishable from a partial one (walking pages until a miss always
    # ends on a miss). Runs in the executor: it does file I/O per page.
    def _load_cached_pages() -> list[PageResult] | None:
        count = cache.get_page_count(pdf_hash)
        if not count:
            return None
        result: list[PageResult] = []
        for page_number in range(1, count + 1):
            key = page_key(pdf_hash, page_number)
            doc = cache.get(key)
            if doc is None or not cache.has_excel(key):
                return None
            result.append(PageResult(page_number=page_number,
                                     excel_filename=f"{key}.xlsx", document=doc))
        return result

    cached_pages = await loop.run_in_executor(None, _load_cached_pages)
    if cached_pages:
        log.info("pdf cache hit", fields={"pdf_hash": pdf_hash,
                                          "pages": len(cached_pages)})
        REQUESTS.labels(type="pdf", outcome="cache_hit").inc()
        return PdfProcessResponse(cache_hit=True, pdf_hash=pdf_hash,
                                  page_count=len(cached_pages), pages=cached_pages)

    # --- Full processing, page by page -------------------------------------
    # The entire blocking pipeline — OCR, Ollama, workbook serialization and
    # cache writes for up to pdf_max_pages pages — runs in a thread so the
    # event loop (and /health) stays responsive throughout.
    def _process_and_store() -> list[PageResult]:
        result: list[PageResult] = []
        for num, doc in process_pdf_bytes(data, settings):
            key = page_key(pdf_hash, num)
            write_workbook(doc, cache.excel_path(key), settings)
            cache.put(key, doc)
            result.append(PageResult(page_number=num,
                                     excel_filename=f"{key}.xlsx", document=doc))
            REGIONS_DETECTED.labels(type="pdf").observe(doc.region_count)
            FIELDS_EXTRACTED.labels(type="pdf").observe(len(doc.extra_fields))
        # Written last: its presence certifies that every page above exists.
        cache.put_page_count(pdf_hash, len(result))
        return result

    t0 = time.perf_counter()
    pages = await loop.run_in_executor(None, _process_and_store)
    PROCESSING_SECONDS.labels(type="pdf").observe(time.perf_counter() - t0)
    REQUESTS.labels(type="pdf", outcome="success").inc()
    log.info("pdf processed", fields={"pdf_hash": pdf_hash, "pages": len(pages)})
    return PdfProcessResponse(cache_hit=False, pdf_hash=pdf_hash,
                              page_count=len(pages), pages=pages)


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
