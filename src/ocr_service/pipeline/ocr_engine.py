"""PaddleOCR engine wrapper.

Wraps PaddleOCR so the rest of the pipeline depends only on our `Region`
schema, not on Paddle's raw output shape (which has changed across versions).
The model is loaded lazily and cached, so importing this module is cheap and
unit tests that don't touch real OCR don't pay the model-load cost.
"""
from __future__ import annotations

import numpy as np

from .config import Settings, get_settings
from .model_loader import get_paddle
from .schemas import RecognizerSource, Region
from .script_detect import detect_script


def _normalize_paddle_result(raw) -> list[Region]:
    """Convert PaddleOCR output into a list of `Region` objects.

    PaddleOCR (.ocr) returns, per image, a list of
        [ [polygon, (text, confidence)], ... ]
    Newer versions nest results per page; we defensively unwrap one level.
    """
    regions: list[Region] = []
    if not raw:
        return regions

    page = raw[0] if (len(raw) == 1 and isinstance(raw[0], list)) else raw
    for entry in page:
        try:
            polygon, (text, conf) = entry[0], entry[1]
        except (ValueError, TypeError, IndexError):
            continue
        poly = [(float(x), float(y)) for x, y in polygon]
        text = (text or "").strip()
        if not text:
            continue
        regions.append(
            Region(
                text=text,
                confidence=float(conf),
                polygon=poly,
                script=detect_script(text),
                source=RecognizerSource.PADDLE,
                paddle_text=text,
                paddle_confidence=float(conf),
            )
        )
    return regions


def run_ocr(image: np.ndarray, settings: Settings | None = None) -> list[Region]:
    """Run PaddleOCR on a (pre-processed) image and return Regions."""
    settings = settings or get_settings()
    ocr = get_paddle(settings)
    raw = ocr.ocr(image, cls=True)
    return _normalize_paddle_result(raw)
