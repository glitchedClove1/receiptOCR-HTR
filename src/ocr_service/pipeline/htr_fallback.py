"""Confidence-based HTR fallback (TrOCR) + script-aware recognizer registry.

Design: a registry maps a `Script` to a recognizer callable. Today only Latin
has a fallback recognizer (TrOCR). Arabic intentionally has none — handwritten
Arabic is out of scope — so low-confidence Arabic regions are flagged and the
Paddle result is kept. To add Arabic HTR later, implement a recognizer and
register it under `Script.ARABIC`; no other code changes are required.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .config import Settings, get_settings
from .model_loader import get_trocr
from .schemas import RecognizerSource, Region, Script

# A recognizer takes a cropped region image and returns recognized text.
Recognizer = Callable[[np.ndarray], str]


def _crop_region(image: np.ndarray, region: Region) -> np.ndarray | None:
    """Crop the axis-aligned bounding box of a region, with a small margin."""
    x0, y0, x1, y1 = region.bbox
    h, w = image.shape[:2]
    pad = 4
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return image[y0:y1, x0:x1]


def _trocr_recognizer_factory(settings: Settings) -> Recognizer:
    """Build a Latin TrOCR recognizer closure bound to the loaded model."""

    def _recognize(crop: np.ndarray) -> str:
        from PIL import Image

        processor, model = get_trocr(settings)
        # TrOCR expects an RGB PIL image.
        if crop.ndim == 2:
            pil = Image.fromarray(crop).convert("RGB")
        else:
            pil = Image.fromarray(crop[:, :, ::-1]).convert("RGB")  # BGR->RGB
        pixel_values = processor(images=pil, return_tensors="pt").pixel_values
        if settings.use_gpu:
            try:
                pixel_values = pixel_values.to("cuda")
            except Exception:
                pass
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.strip()

    return _recognize


def build_registry(settings: Settings) -> dict[Script, Recognizer]:
    """Return the script -> recognizer registry.

    Extend here to add new scripts (e.g. register `Script.ARABIC`).
    """
    return {
        Script.LATIN: _trocr_recognizer_factory(settings),
        # Script.ARABIC: <register an Arabic HTR recognizer here in future>
    }


def apply_fallback(
    image: np.ndarray,
    regions: list[Region],
    settings: Settings | None = None,
    registry: dict[Script, Recognizer] | None = None,
) -> list[Region]:
    """Re-run low-confidence regions through the appropriate HTR recognizer.

    Decision rule (documented):
      * If Paddle confidence >= threshold -> keep Paddle result untouched.
      * If below threshold and a recognizer exists for the region's script ->
        run it, store both candidates, and adopt the fallback's text.
      * If below threshold but no recognizer exists (e.g. Arabic) -> keep the
        Paddle result and flag `low_confidence_arabic` for human review.
    """
    settings = settings or get_settings()
    registry = registry if registry is not None else build_registry(settings)
    threshold = settings.confidence_threshold

    for region in regions:
        if region.confidence >= threshold:
            continue

        recognizer = registry.get(region.script)
        if recognizer is None:
            if region.script == Script.ARABIC:
                region.low_confidence_arabic = True
            continue

        crop = _crop_region(image, region)
        if crop is None:
            region.fallback_failed = True
            continue
        try:
            trocr_text = recognizer(crop)
        except Exception:
            # Graceful degradation: a single failed region must not abort —
            # but the kept low-confidence text must be flagged for review.
            region.fallback_failed = True
            continue

        region.trocr_text = trocr_text
        if trocr_text:
            region.text = trocr_text
            region.source = RecognizerSource.TROCR
        else:
            # Fallback produced nothing usable; the Paddle text stays but is
            # below threshold and unverified.
            region.fallback_failed = True

    return regions
