"""Thread-safe lazy loaders for the heavy OCR/HTR models.

Models are expensive to construct and must be created exactly once even under
concurrent requests. Each loader uses double-checked locking: the fast path is
a lockless check of the cached instance; only the first caller(s) take the lock
and build the model. A failed load raises ModelLoadError rather than caching a
broken half-initialized object.
"""
from __future__ import annotations

import os
import threading
from typing import Any

from ..exceptions import ModelLoadError
from ..logging_config import get_logger
from .config import Settings

log = get_logger(__name__)

_paddle_instance: Any | None = None
_paddle_lock = threading.Lock()

_trocr_processor: Any | None = None
_trocr_model: Any | None = None
_trocr_lock = threading.Lock()


def get_paddle(settings: Settings) -> Any:
    """Return a shared PaddleOCR instance, constructing it once if needed."""
    global _paddle_instance
    if _paddle_instance is not None:  # fast path, no lock
        return _paddle_instance
    with _paddle_lock:
        if _paddle_instance is not None:  # re-check under lock
            return _paddle_instance
        try:
            from paddleocr import PaddleOCR

            # Disable oneDNN/MKL-DNN at the environment level — the PaddlePaddle
            # CPU executor crashes on some Windows builds when it falls back from
            # a failed GPU init while oneDNN is active.
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

            log.info("loading paddleocr", fields={"lang": settings.paddle_primary_lang})
            _paddle_instance = PaddleOCR(
                use_textline_orientation=True,
                lang=settings.paddle_primary_lang,
                device="gpu" if settings.use_gpu else "cpu",
                enable_mkldnn=False,
            )
            return _paddle_instance
        except Exception as exc:  # noqa: BLE001 - re-raised as domain error
            raise ModelLoadError(f"PaddleOCR failed to load: {exc}") from exc


def get_trocr(settings: Settings) -> tuple[Any, Any]:
    """Return a shared (processor, model) TrOCR pair, constructing once."""
    global _trocr_processor, _trocr_model
    if _trocr_model is not None and _trocr_processor is not None:
        return _trocr_processor, _trocr_model
    with _trocr_lock:
        if _trocr_model is not None and _trocr_processor is not None:
            return _trocr_processor, _trocr_model
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            log.info("loading trocr", fields={"model": settings.trocr_model_name})
            processor = TrOCRProcessor.from_pretrained(settings.trocr_model_name)
            model = VisionEncoderDecoderModel.from_pretrained(settings.trocr_model_name)
            if settings.use_gpu:
                try:
                    model = model.to("cuda")
                except Exception:  # noqa: BLE001 - GPU optional, fall back to CPU
                    log.warning("trocr gpu move failed; using cpu")
            model.eval()
            _trocr_processor, _trocr_model = processor, model
            return _trocr_processor, _trocr_model
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"TrOCR failed to load: {exc}") from exc


def reset_models() -> None:
    """Clear cached models (used by tests)."""
    global _paddle_instance, _trocr_processor, _trocr_model
    _paddle_instance = None
    _trocr_processor = None
    _trocr_model = None
