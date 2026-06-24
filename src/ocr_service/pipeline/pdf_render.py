"""PDF rasterization (PyMuPDF / fitz).

Renders each PDF page to an OpenCV BGR image so it can flow through the exact
same preprocess -> OCR -> extract pipeline as a regular image upload. Per the
chosen design, every page is rasterized and OCR'd (no embedded-text shortcut)
for consistent behaviour across digital and scanned PDFs.

PyMuPDF is AGPL-3.0; acceptable for internal/hosted use. The import is local
so the dependency only loads when a PDF is actually processed.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from ..exceptions import InvalidPdfError
from ..logging_config import get_logger
from .config import Settings, get_settings

log = get_logger(__name__)


def _pixmap_to_bgr(pix) -> np.ndarray:
    """Convert a PyMuPDF Pixmap to an OpenCV BGR ndarray."""
    # pix.samples is a bytes buffer of shape (h, w, n) where n is 1/3/4.
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 1:  # grayscale -> BGR
        return np.repeat(arr, 3, axis=2)
    if pix.n == 4:  # RGBA -> drop alpha, RGB -> BGR
        arr = arr[:, :, :3]
    # PyMuPDF gives RGB; OpenCV downstream expects BGR.
    return arr[:, :, ::-1].copy()


def iter_pdf_pages(
    data: bytes, settings: Settings | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (page_number, bgr_image) for each page, 1-indexed.

    Raises InvalidPdfError if the bytes are not a readable PDF. Honors the
    configured DPI and the page cap.
    """
    settings = settings or get_settings()

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - dependency presence
        raise InvalidPdfError(
            "PDF support requires PyMuPDF; install the '[ocr]' extra."
        ) from exc

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001
        raise InvalidPdfError(f"Could not open PDF: {exc}") from exc

    try:
        total = doc.page_count
        if total == 0:
            raise InvalidPdfError("PDF has no pages.")
        limit = min(total, settings.pdf_max_pages)
        if total > limit:
            log.warning(
                "pdf page cap applied",
                fields={"total_pages": total, "processed_pages": limit},
            )
        # A zoom factor maps DPI to PyMuPDF's 72-dpi default coordinate space.
        zoom = settings.pdf_render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)

        for index in range(limit):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            yield index + 1, _pixmap_to_bgr(pix)
    finally:
        doc.close()


def count_pages(data: bytes) -> int:
    """Return the number of pages in a PDF (for diagnostics/limits)."""
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count
