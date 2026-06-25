"""PaddleOCR engine wrapper.

Wraps PaddleOCR so the rest of the pipeline depends only on our `Region`
schema, not on Paddle's raw output shape (which has changed across versions).
The model is loaded lazily and cached, so importing this module is cheap and
unit tests that don't touch real OCR don't pay the model-load cost.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Settings, get_settings
from .model_loader import get_paddle
from .schemas import RecognizerSource, Region
from .script_detect import detect_script


def _normalize_paddle_result(raw) -> list[Region]:
    regions: list[Region] = []
    if not raw:
        return regions

    for res in raw:
        def _get(key):
            if isinstance(res, dict):
                return res.get(key)
            try:
                return res[key]
            except (TypeError, KeyError, IndexError):
                return getattr(res, key, None)

        texts = _get("rec_texts") or []
        scores = _get("rec_scores") or []
        polys = _get("rec_polys")
        if polys is None:
            polys = _get("dt_polys") or []

        for text, conf, polygon in zip(texts, scores, polys):
            text = (text or "").strip()
            if not text:
                continue
            poly = []
            try:
                for point in polygon:
                    poly.append((float(point[0]), float(point[1])))
            except (TypeError, ValueError, IndexError):
                poly = []
            if not poly:
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
    # PaddleOCR 3.x requires a 3-channel image; preprocessing always yields grayscale.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    raw = list(ocr.predict(image))   # predict() is a generator in 3.x — materialize it
    return _normalize_paddle_result(raw)
