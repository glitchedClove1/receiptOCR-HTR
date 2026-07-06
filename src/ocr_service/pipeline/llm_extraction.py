"""Full-document field extraction via a local Ollama model.

Two extraction paths are tried in order:

1. **VLM path** (``vlm_model`` setting, default ``qwen2.5vl:3b``)
   The preprocessed image is sent as a base64 JPEG directly to the model.
   The model sees the raw pixel data — no OCR errors can corrupt its input.

2. **Text path** (``ollama_model`` setting, default ``qwen2.5:3b``)
   Spatially-ordered OCR text is sent to a text-only model. Used when the
   VLM is unavailable, times out, or ``vlm_model`` is left empty.

Fallback strategy
-----------------
Each path caches the last model that worked (``_working_vlm_model`` and
``_working_model``). On the first request the configured model is tried; on
HTTP 500 (GPU VRAM overflow) the next smaller model in the family chain is
tried automatically. Caches reset on service restart.

If all models fail the function returns an empty dict — the pipeline is
never blocked.
"""
from __future__ import annotations

import base64
import json

import cv2
import numpy as np
import requests

from ..logging_config import get_logger
from .config import Settings
from .schemas import FieldValue, Region

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model fallback chains
# ---------------------------------------------------------------------------

_VLM_FALLBACK_CHAIN: list[str] = [
    "qwen2.5vl:72b",
    "qwen2.5vl:32b",
    "qwen2.5vl:7b",
    "qwen2.5vl:3b",
    "moondream",
]

_TEXT_FALLBACK_CHAIN: list[str] = [
    "qwen2.5:72b",
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "qwen2.5:0.5b",
]

_working_vlm_model: str | None = None
_working_model: str | None = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_VLM_PROMPT = """\
You are digitizing a scanned financial document. Look at the image and extract \
every piece of information that has an actual value.

Return a FLAT JSON object — no nested objects, no arrays.
Each key is a short descriptive English field name.
Each value is a non-empty string taken directly from the document.
OMIT any field whose value is empty, unknown, or not visible.
Do not invent values. Do not include placeholder or template fields.

Examples of what to extract when present:
Vendor Name, Vendor Address, Fiscal Matrix Number, CCB Number,
Invoice Number, Invoice Date, Due Date,
Subtotal, VAT Rate, VAT Amount, Grand Total, Currency,
Item Reference, Item Description, Quantity, Unit Price,
Payment Terms, Payment Method, Bank Name, IBAN,
Client Name, Client Address, Settlement Amount"""

_TEXT_PROMPT_TEMPLATE = """\
You are digitizing a scanned financial document. Extract every piece of \
information that has an actual value in the text below.

Return a FLAT JSON object — no nested objects, no arrays.
Each key is a short descriptive English field name.
Each value is a non-empty string taken directly from the document.
OMIT any field whose value is empty, unknown, or not present in the text.
Do not invent values. Do not include placeholder or template fields.

Examples of what to extract when present:
Vendor Name, Vendor Address, Fiscal Matrix Number, CCB Number,
Invoice Number, Invoice Date, Due Date,
Subtotal, VAT Rate, VAT Amount, Grand Total, Currency,
Item Reference, Item Description, Quantity, Unit Price,
Payment Terms, Payment Method, Bank Name, IBAN,
Client Name, Client Address, Settlement Amount

Document text:
{text}"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fallback_chain_from(primary: str, chain: list[str]) -> list[str]:
    """Return [primary, next-smaller, ...] from the given family chain."""
    if primary in chain:
        return chain[chain.index(primary):]
    return [primary]


def _flatten(obj: object, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict into a flat {label: value} mapping."""
    result: dict[str, str] = {}
    if not isinstance(obj, dict):
        return result
    for k, v in obj.items():
        if not isinstance(k, str):
            continue
        label = f"{prefix} {k}".strip() if prefix else k.strip()
        if isinstance(v, dict):
            result.update(_flatten(v, label))
        elif isinstance(v, (str, int, float)):
            val = str(v).strip()
            if val:
                result[label] = val
    return result


def _rows_to_text(rows: list[list[Region]]) -> str:
    """Flatten spatially ordered OCR rows into a readable string."""
    lines: list[str] = []
    for row in rows:
        line = " ".join(r.text for r in row).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


_VLM_MAX_SIDE = 1024


def _encode_image(image: np.ndarray) -> str:
    """Encode a numpy image array to a base64 JPEG string for Ollama.

    Downscales the image so its longest side is at most _VLM_MAX_SIDE pixels
    before encoding. High-DPI scans can otherwise exceed the model's context
    window just from image tokens alone.
    """
    if image.ndim == 2:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = image
    h, w = img_bgr.shape[:2]
    if max(h, w) > _VLM_MAX_SIDE:
        scale = _VLM_MAX_SIDE / max(h, w)
        img_bgr = cv2.resize(
            img_bgr,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode()


def _parse_response(resp: requests.Response, model: str) -> dict | None:
    """Extract and parse the JSON dict from an Ollama generate response."""
    if resp.status_code != 200:
        log.warning("ollama_model_error", fields={
            "model": model,
            "status": resp.status_code,
            "body": resp.text[:200],
        })
        return None
    try:
        raw = resp.json().get("response", "{}")
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        log.warning("ollama_json_parse_error", fields={"model": model, "error": str(exc)})
        return None
    if not isinstance(parsed, dict):
        log.warning("ollama_unexpected_shape", fields={
            "model": model, "type": type(parsed).__name__,
        })
        return None
    return parsed


def _try_vlm_model(
    model: str,
    image_b64: str,
    base_url: str,
    timeout: int,
) -> dict | None:
    """Send image to a VLM and return the parsed JSON dict, or None on failure."""
    payload = {
        "model": model,
        "prompt": _VLM_PROMPT,
        "images": [image_b64],
        "format": "json",
        "stream": False,
        # Images consume many tokens; 8192 comfortably fits a full invoice
        # image + prompt without hitting the default 4096 token context limit.
        "options": {"temperature": 0.0, "num_ctx": 8192},
    }
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        log.warning("ollama_unavailable", fields={"url": base_url})
        return None
    except requests.exceptions.Timeout:
        log.warning("ollama_timeout", fields={"model": model, "timeout": timeout})
        return None
    except requests.exceptions.RequestException as exc:
        log.warning("ollama_request_error", fields={"model": model, "error": str(exc)})
        return None
    return _parse_response(resp, model)


def _try_text_model(
    model: str,
    prompt: str,
    base_url: str,
    timeout: int,
) -> dict | None:
    """Send OCR text to a text model and return the parsed JSON dict, or None."""
    payload = {
        "model": model,
        "prompt": prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=timeout,
        )
    except requests.exceptions.ConnectionError:
        log.warning("ollama_unavailable", fields={"url": base_url})
        return None
    except requests.exceptions.Timeout:
        log.warning("ollama_timeout", fields={"model": model, "timeout": timeout})
        return None
    except requests.exceptions.RequestException as exc:
        log.warning("ollama_request_error", fields={"model": model, "error": str(exc)})
        return None
    return _parse_response(resp, model)


def _fields_from_parsed(parsed: dict, model: str) -> dict[str, FieldValue]:
    result: dict[str, FieldValue] = {}
    for label, val in _flatten(parsed).items():
        if label and val:
            result[label] = FieldValue(value=val, corrected=False, needs_review=False)
    log.info("ollama_fields_extracted", fields={"model": model, "count": len(result)})
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def extract_extra_fields(
    rows: list[list[Region]],
    settings: Settings,
    image: np.ndarray | None = None,
) -> dict[str, FieldValue]:
    """Extract every labeled field from the document via the local LLM.

    Tries the VLM (image input) first when an image is provided, then falls
    back to the text model using OCR output. Returns an empty dict if all
    models fail so the pipeline is never blocked.
    """
    global _working_vlm_model, _working_model

    if not settings.ollama_enabled:
        return {}

    base_url = settings.ollama_base_url
    timeout = settings.ollama_timeout_seconds

    # --- VLM path (image → model) -------------------------------------------
    if image is not None and settings.vlm_model:
        try:
            image_b64 = _encode_image(image)
        except Exception as exc:  # noqa: BLE001
            log.warning("vlm_encode_failed", fields={"error": str(exc)})
            image_b64 = None

        if image_b64 is not None:
            chain = _fallback_chain_from(settings.vlm_model, _VLM_FALLBACK_CHAIN)
            if _working_vlm_model in chain:
                chain = chain[chain.index(_working_vlm_model):]

            for model in chain:
                parsed = _try_vlm_model(model, image_b64, base_url, timeout)
                if parsed is None:
                    continue
                if model != settings.vlm_model:
                    log.warning("vlm_using_fallback", fields={
                        "configured": settings.vlm_model, "using": model,
                    })
                if _working_vlm_model != model:
                    log.info("vlm_working_model_set", fields={"model": model})
                    _working_vlm_model = model
                return _fields_from_parsed(parsed, model)

            log.warning("vlm_all_models_failed", fields={"chain": chain})

    # --- Text path fallback (OCR text → text model) -------------------------
    text = _rows_to_text(rows)
    if not text.strip():
        return {}

    prompt = _TEXT_PROMPT_TEMPLATE.format(text=text)
    chain = _fallback_chain_from(settings.ollama_model, _TEXT_FALLBACK_CHAIN)
    if _working_model in chain:
        chain = chain[chain.index(_working_model):]

    for model in chain:
        parsed = _try_text_model(model, prompt, base_url, timeout)
        if parsed is None:
            continue
        if model != settings.ollama_model:
            log.warning("ollama_using_fallback", fields={
                "configured": settings.ollama_model, "using": model,
            })
        if _working_model != model:
            log.info("ollama_working_model_set", fields={"model": model})
            _working_model = model
        return _fields_from_parsed(parsed, model)

    log.warning("ollama_all_models_failed", fields={"chain": chain})
    return {}
