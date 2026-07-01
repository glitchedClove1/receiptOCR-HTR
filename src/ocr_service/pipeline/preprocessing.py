"""Image pre-processing (OpenCV).

Each step is an independently callable function so they can be toggled or
reordered via config. `preprocess` orchestrates them and returns the cleaned
image plus the ordered list of operations actually applied.
"""
from __future__ import annotations

import cv2
import numpy as np

from .config import Settings, get_settings


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a BGR/colour image to single-channel grayscale."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization.

    CLAHE works well on documents with uneven lighting (e.g. phone photos
    of receipts) where global equalization would blow out highlights.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def remove_noise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise.

    fastNlMeans removes sensor/scan grain while keeping stroke edges, which
    matters for thin handwriting that bilateral filtering can smear.
    """
    return cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)


_MAX_SKEW_DEGREES = 15.0


def _detect_skew_angle(gray: np.ndarray) -> float:
    """Estimate document skew in degrees via the minAreaRect of dark pixels.

    Returns a small correction angle in the range (-15, +15) degrees.
    minAreaRect on the full page can return angles near ±90° when it locks
    onto the page bounding box rather than text line orientation — those are
    clamped to 0 so we never apply a catastrophic rotation to a good image.
    """
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.shape[0] < 50:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    # OpenCV returns angle in [-90, 0); normalize to a small correction.
    if angle < -45:
        angle = 90 + angle
    # Discard implausible angles — they indicate a bad minAreaRect estimate.
    if abs(angle) > _MAX_SKEW_DEGREES:
        return 0.0
    return -angle


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the image to straighten skewed text. Returns (image, angle)."""
    angle = _detect_skew_angle(gray)
    if abs(angle) < 0.1:  # nothing meaningful to correct
        return gray, 0.0
    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, angle


def preprocess(
    image: np.ndarray, settings: Settings | None = None
) -> tuple[np.ndarray, list[str]]:
    """Run the configured preprocessing chain.

    Returns the processed image and a record of operations applied (useful
    for diagnostics and for reproducing results). Steps are individually
    toggleable through settings.
    """
    settings = settings or get_settings()
    applied: list[str] = []
    out = image

    if settings.pre_grayscale:
        out = to_grayscale(out)
        applied.append("grayscale")
    else:
        # Downstream steps assume single channel; force it regardless.
        out = to_grayscale(out)

    if settings.pre_clahe:
        out = enhance_contrast(out)
        applied.append("clahe")

    if settings.pre_denoise:
        out = remove_noise(out)
        applied.append("denoise")

    if settings.pre_deskew:
        out, angle = deskew(out)
        if angle != 0.0:
            applied.append(f"deskew({angle:.2f}deg)")

    return out, applied
