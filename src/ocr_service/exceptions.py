"""Domain exception hierarchy.

Distinguishing failure modes lets the API map each to the right HTTP status
and lets logs/metrics tell apart "the user sent a bad image" from "a model
failed to load" — which matter very differently operationally.
"""
from __future__ import annotations


class OCRServiceError(Exception):
    """Base class for all service errors."""


class InvalidImageError(OCRServiceError):
    """The uploaded bytes could not be decoded as an image (client fault)."""


class InvalidPdfError(OCRServiceError):
    """The uploaded bytes could not be opened as a PDF (client fault)."""


class UnsupportedMediaError(OCRServiceError):
    """The upload's content type is not an accepted image type (client fault)."""


class PayloadTooLargeError(OCRServiceError):
    """The upload exceeds the configured size limit (client fault)."""


class ModelLoadError(OCRServiceError):
    """An OCR/HTR model failed to load (server/ops fault, not the user's)."""


class PipelineError(OCRServiceError):
    """A stage failed unexpectedly while processing a valid image."""
