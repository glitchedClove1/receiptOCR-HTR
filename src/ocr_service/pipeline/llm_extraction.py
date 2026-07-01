"""Full-document field extraction via a local Ollama LLM.

The LLM reads spatially-ordered OCR text and extracts every labeled piece of
information visible on the page, exactly as a human would when scanning a
physical document.

Fallback strategy
-----------------
A module-level ``_working_model`` caches the last model that responded
successfully within this process lifetime. On the first request, the
configured model is tried. If it returns HTTP 500 (e.g. GPU VRAM overflow)
or times out, the next smaller model in ``_FALLBACK_CHAIN`` is tried
automatically. Once a working model is found it is reused for every subsequent
call — no wasted GPU crash on every request. Restarting the service resets
the cache, giving the configured model another chance (useful after a CUDA fix
or model swap).

If Ollama is unavailable, all models fail, or JSON cannot be parsed, the
function returns an empty dict so the pipeline is never blocked.
"""
from __future__ import annotations

import json

import requests

from ..logging_config import get_logger
from .config import Settings
from .schemas import FieldValue, Region

log = get_logger(__name__)

# Models tried in order when the primary crashes or times out.
# Covers the full qwen2.5 family; other Ollama models can be appended.
_FALLBACK_CHAIN: list[str] = [
    "qwen2.5:72b",
    "qwen2.5:32b",
    "qwen2.5:14b",
    "qwen2.5:7b",
    "qwen2.5:3b",
    "qwen2.5:1.5b",
    "qwen2.5:0.5b",
]

# Last model that produced a successful response in this process.
# None means "not yet determined" — try from the configured model.
_working_model: str | None = None

_PROMPT_TEMPLATE = """\
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


def _fallback_chain_from(primary: str) -> list[str]:
    """Return [primary, next-smaller, ...] using the known family list.

    If the primary is not in the list it is still tried first, with no
    automatic fallbacks beyond it.
    """
    if primary in _FALLBACK_CHAIN:
        return _FALLBACK_CHAIN[_FALLBACK_CHAIN.index(primary):]
    return [primary]


def _flatten(obj: object, prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested dict into a flat {label: value} mapping.

    Handles whatever nesting structure the LLM decides to return so we never
    silently drop fields that came back as sub-objects. Empty values excluded.
    """
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


def _try_model(
    model: str,
    prompt: str,
    base_url: str,
    timeout: int,
) -> dict | None:
    """Attempt one generate call. Returns the parsed JSON dict or None.

    None covers: HTTP error (including 500 GPU crash), network failure,
    timeout, and invalid JSON — all treated identically as "try next model".
    """
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


def extract_extra_fields(
    rows: list[list[Region]],
    settings: Settings,
) -> dict[str, FieldValue]:
    """Transcribe every labeled field from the document via the local LLM.

    Tries the configured model first; on failure steps down the fallback chain
    to the next smaller model. Caches the working model for the lifetime of
    this process so subsequent calls skip known-broken models immediately.

    Returns an empty dict if all models fail.
    """
    global _working_model

    if not settings.ollama_enabled:
        return {}

    text = _rows_to_text(rows)
    if not text.strip():
        return {}

    prompt = _PROMPT_TEMPLATE.format(text=text)
    chain = _fallback_chain_from(settings.ollama_model)

    # If we already know a working model, skip everything above it in the chain.
    if _working_model in chain:
        chain = chain[chain.index(_working_model):]

    for model in chain:
        parsed = _try_model(
            model, prompt, settings.ollama_base_url, settings.ollama_timeout_seconds
        )
        if parsed is None:
            continue

        # This model worked — cache it and move on.
        if model != settings.ollama_model:
            log.warning("ollama_using_fallback", fields={
                "configured": settings.ollama_model,
                "using": model,
            })
        if _working_model != model:
            log.info("ollama_working_model_set", fields={"model": model})
            _working_model = model

        result: dict[str, FieldValue] = {}
        for label, val in _flatten(parsed).items():
            if label and val:
                result[label] = FieldValue(value=val, corrected=False, needs_review=False)

        log.info("ollama_fields_extracted", fields={"model": model, "count": len(result)})
        return result

    log.warning("ollama_all_models_failed", fields={"chain": chain})
    return {}
