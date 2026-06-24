"""Pipeline orchestrator.

Ties the stages together: decode -> preprocess -> OCR -> HTR fallback ->
correction -> extraction -> Document. The FastAPI layer wraps this with
caching and Excel generation. Each stage is wrapped so a failure in one
region/field degrades gracefully rather than aborting the document.
"""
from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from ..exceptions import InvalidImageError, ModelLoadError, PipelineError
from ..logging_config import get_logger
from .config import Settings, get_settings
from .correction import correct_text
from .extraction import extract_document
from .htr_fallback import apply_fallback
from .ocr_engine import run_ocr
from .pdf_render import iter_pdf_pages
from .preprocessing import preprocess
from .schemas import Document, Region

log = get_logger(__name__)


def decode_image(data: bytes) -> np.ndarray:
    """Decode raw bytes into an OpenCV BGR image."""
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageError("Could not decode image bytes.")
    return image


def _apply_corrections(regions: list[Region]) -> None:
    """Run numeric-context correction across all region texts in place."""
    for region in regions:
        corrected, changed = correct_text(region.text)
        if changed:
            region.text = corrected
            region.corrected = True


def process_image_array(
    image: np.ndarray, settings: Settings | None = None
) -> Document:
    """Full pipeline on an already-decoded image array -> Document.

    This is the reusable core: image-file uploads decode to an array first,
    and each rasterized PDF page is fed here directly.
    """
    settings = settings or get_settings()

    processed, applied = preprocess(image, settings)

    try:
        # OCR — PaddleOCR returns regions with confidence.
        regions = run_ocr(processed, settings)
        # HTR fallback on low-confidence Latin regions (Arabic flagged, not run).
        regions = apply_fallback(processed, regions, settings)
    except ModelLoadError:
        raise  # let the API map this to 503 distinctly from a generic failure
    except Exception as exc:  # noqa: BLE001 - any other stage failure -> 500
        log.error("ocr stage failed", fields={"error": str(exc)})
        raise PipelineError(str(exc)) from exc

    # Numeric correction (pure-Python, low risk).
    _apply_corrections(regions)

    # Structured extraction.
    try:
        doc: Document = extract_document(regions, settings)
    except Exception as exc:  # noqa: BLE001
        log.error("extraction failed", fields={"error": str(exc)})
        raise PipelineError(str(exc)) from exc

    doc.preprocessing_applied = applied
    return doc


def process_image_bytes(
    data: bytes, settings: Settings | None = None
) -> Document:
    """Full pipeline on raw image bytes -> structured Document."""
    settings = settings or get_settings()
    image = decode_image(data)  # raises InvalidImageError
    return process_image_array(image, settings)


def process_pdf_bytes(
    data: bytes, settings: Settings | None = None
) -> Iterator[tuple[int, Document]]:
    """Process each PDF page independently, yielding (page_number, Document).

    Per the chosen design, each page is a separate document with its own
    workbook. Pages are rasterized and OCR'd individually and reuse the exact
    same core pipeline as image uploads. A failure on one page raises and is
    surfaced by the caller; pages already yielded remain valid.
    """
    settings = settings or get_settings()
    for page_number, image in iter_pdf_pages(data, settings):
        doc = process_image_array(image, settings)
        doc.page_number = page_number
        yield page_number, doc
