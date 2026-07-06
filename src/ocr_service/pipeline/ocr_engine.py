"""PaddleOCR engine wrapper.

Wraps PaddleOCR so the rest of the pipeline depends only on our `Region`
schema, not on Paddle's raw output shape (which has changed across versions).
The model is loaded lazily and cached, so importing this module is cheap and
unit tests that don't touch real OCR don't pay the model-load cost.
"""
from __future__ import annotations

import threading

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

    def _get(res, key):
        if isinstance(res, dict):
            return res.get(key)
        try:
            return res[key]
        except (TypeError, KeyError, IndexError):
            return getattr(res, key, None)

    for res in raw:
        # Explicit None checks: these values can be numpy arrays, whose
        # truthiness (`x or []`) raises "ambiguous truth value".
        texts = _get(res, "rec_texts")
        texts = list(texts) if texts is not None else []
        scores = _get(res, "rec_scores")
        scores = list(scores) if scores is not None else []
        polys = _get(res, "rec_polys")
        if polys is None:
            polys = _get(res, "dt_polys")
        polys = list(polys) if polys is not None else []

        for text, conf, polygon in zip(texts, scores, polys, strict=False):
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


# Paddle inference predictors hold mutable internal state and are not
# thread-safe. Requests run in a thread pool, so inference on the single
# shared instance must be serialized.
_predict_lock = threading.Lock()


def run_ocr(image: np.ndarray, settings: Settings | None = None) -> list[Region]:
    """Run PaddleOCR on a (pre-processed) image and return Regions."""
    settings = settings or get_settings()
    ocr = get_paddle(settings)
    # PaddleOCR 3.x requires a 3-channel image; preprocessing always yields grayscale.
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    with _predict_lock:
        raw = list(ocr.predict(image))   # predict() is a generator in 3.x
    return _normalize_paddle_result(raw)
