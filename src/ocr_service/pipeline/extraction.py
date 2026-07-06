"""Structured field extraction (rule-based + spatial heuristics).

No LLM, no API. We group regions into visual rows by vertical proximity, then
apply regex/keyword rules plus left-to-right spatial reasoning to pull header
fields, a line-item table, and totals. Every emitted field carries provenance
(source engine, confidence, whether corrected) and a `needs_review` flag.
"""
from __future__ import annotations

import re

import numpy as np

from .config import Settings, get_settings
from .correction import (
    correct_text,
    normalize_date,
    parse_currency,
    parse_number,
)
from .llm_extraction import extract_extra_fields
from .schemas import (
    Document,
    FieldValue,
    LineItem,
    RecognizerSource,
    Region,
)

# Keyword cues (FR + EN). Lowercased substring match.
_TOTAL_CUES = ("total ttc", "grand total", "total a payer", "net a payer",
               "total", "montant total")
_SUBTOTAL_CUES = ("sous-total", "sous total", "subtotal", "total ht", "ht")
_TAX_CUES = ("tva", "vat", "tax", "taxe")
_INVOICE_CUES = ("facture", "invoice", "recu", "reçu", "ticket", "n°", "no.", "no ")
_TAXID_CUES = ("matricule fiscal", "tax id", "vat number", "siret", "if ",
               "mf ", "rc ")
_DATE_CUES = ("date", "le ")

_QTY_RE = re.compile(r"^\s*(\d{1,3})\s*[xX]?\s*$")
_MONEY_RE = re.compile(r"\d+[.,]\d{2}")


def _adaptive_y_tol(regions: list[Region], fallback: float = 12.0) -> float:
    """Compute a row-grouping tolerance from the median region height.

    Using 55% of the median text-line height keeps same-row regions together
    while separating adjacent rows, regardless of scan resolution or font size.
    """
    if not regions:
        return fallback
    heights = []
    for r in regions:
        ys = [p[1] for p in r.polygon]
        heights.append(max(ys) - min(ys))
    median_h = sorted(heights)[len(heights) // 2]
    return max(fallback, median_h * 0.55)


def _group_rows(regions: list[Region], y_tol: float | None = None) -> list[list[Region]]:
    """Group regions into visual rows by vertical centre proximity.

    Tolerance is derived from the median region height when not specified,
    so the grouping scales correctly across scan resolutions and font sizes.
    """
    tol = y_tol if y_tol is not None else _adaptive_y_tol(regions)
    rows: list[list[Region]] = []
    for region in sorted(regions, key=lambda r: r.center_y):
        placed = False
        for row in rows:
            if abs(row[0].center_y - region.center_y) <= tol:
                row.append(region)
                placed = True
                break
        if not placed:
            rows.append([region])
    for row in rows:
        row.sort(key=lambda r: r.center_x)
    return rows


def _field_from_region(region: Region, value: str, corrected: bool) -> FieldValue:
    """Build a FieldValue, flagging review for low-conf or Arabic regions."""
    return FieldValue(
        value=value,
        source=region.source,
        confidence=region.confidence,
        corrected=corrected,
        needs_review=region.low_confidence_arabic
        or region.source == RecognizerSource.TROCR,
    )


def _has_cue(text_lower: str, cues) -> bool:
    return any(cue in text_lower for cue in cues)


def _extract_money_after_cue(row: list[Region]) -> tuple | None:
    """From a row, return (value_float, region) of the rightmost money token."""
    best = None
    for region in row:
        for m in _MONEY_RE.finditer(region.text):
            val = parse_number(m.group(0))
            if val is not None:
                best = (val, region)  # keep last -> rightmost
    return best


def extract_document(
    regions: list[Region],
    settings: Settings | None = None,
    image: np.ndarray | None = None,
) -> Document:
    """Run the full rule-based extraction over recognized regions."""
    settings = settings or get_settings()
    doc = Document()
    doc.region_count = len(regions)
    doc.low_confidence_count = sum(
        1 for r in regions
        if r.low_confidence_arabic or r.source == RecognizerSource.TROCR
    )

    rows = _group_rows(regions)

    # --- Vendor: heuristic = first non-empty row near the top, not a cue. ----
    for row in rows[:3]:
        joined = " ".join(r.text for r in row).strip()
        if joined and not _has_cue(joined.lower(), _TOTAL_CUES + _TAX_CUES):
            doc.vendor = _field_from_region(row[0], joined, False)
            break

    currency_seen: str | None = None

    # --- Scan rows for header fields and totals ------------------------------
    for row in rows:
        joined = " ".join(r.text for r in row)
        low = joined.lower()

        if currency_seen is None:
            cur = parse_currency(joined)
            if cur:
                currency_seen = cur

        # Date
        if doc.document_date.value is None and _has_cue(low, _DATE_CUES):
            iso = normalize_date(joined)
            if iso:
                doc.document_date = _field_from_region(row[0], iso, True)
        if doc.document_date.value is None:  # date may appear without a cue
            iso = normalize_date(joined)
            if iso:
                doc.document_date = _field_from_region(row[0], iso, True)

        # Invoice number
        if doc.invoice_number.value is None and _has_cue(low, _INVOICE_CUES):
            m = re.search(r"(?:n[°o]\.?|no\.?|#)\s*([A-Za-z0-9\-/]+)", joined,
                          re.IGNORECASE)
            if m:
                corrected, ch = correct_text(m.group(1))
                doc.invoice_number = _field_from_region(row[0], corrected, ch)

        # Tax ID
        if doc.vendor_tax_id.value is None and _has_cue(low, _TAXID_CUES):
            m = re.search(r"([A-Z0-9]{6,}[A-Z0-9/\-]*)", joined)
            if m:
                doc.vendor_tax_id = _field_from_region(row[0], m.group(1), False)

        # Totals — order matters: check grand total cues before bare 'total'.
        money = _extract_money_after_cue(row)
        if money:
            val, region = money
            if _has_cue(low, _TAX_CUES) and doc.tax.value is None:
                doc.tax = _field_from_region(region, str(val), True)
            elif _has_cue(low, _SUBTOTAL_CUES) and doc.subtotal.value is None:
                doc.subtotal = _field_from_region(region, str(val), True)
            elif _has_cue(low, _TOTAL_CUES) and doc.grand_total.value is None:
                doc.grand_total = _field_from_region(region, str(val), True)

    if currency_seen:
        doc.currency = FieldValue(value=currency_seen, corrected=False)

    # --- Line items: rows that have a description + at least one money token,
    #     excluding rows that matched totals/tax cues. ------------------------
    doc.line_items = _extract_line_items(rows)

    # --- LLM extraction: transcribe every field visible on the document. -----
    doc.extra_fields = extract_extra_fields(rows, settings, image=image)

    return doc


def _extract_line_items(rows: list[list[Region]]) -> list[LineItem]:
    """Infer line items from rows containing a description and money values.

    Heuristic: a line-item row has leading text (the description) and one or
    two trailing money tokens (unit price and/or line total), optionally a
    leading quantity. Rows whose text hits total/tax/subtotal cues are skipped.
    """
    items: list[LineItem] = []
    skip_cues = _TOTAL_CUES + _TAX_CUES + _SUBTOTAL_CUES

    for row in rows:
        joined = " ".join(r.text for r in row)
        low = joined.lower()
        if _has_cue(low, skip_cues):
            continue

        money_vals = [
            parse_number(m.group(0)) for m in _MONEY_RE.finditer(joined)
        ]
        money_vals = [v for v in money_vals if v is not None]
        if not money_vals:
            continue

        # Description = leftmost regions before the first money token.
        # A standalone small integer among those tokens is taken as quantity,
        # regardless of whether it precedes or follows the description text
        # (receipt layouts vary).
        desc_parts: list[str] = []
        qty: float | None = None
        for region in row:
            if _MONEY_RE.search(region.text):
                break
            qm = _QTY_RE.match(region.text)
            if qm and qty is None:
                qty = float(qm.group(1))
                continue
            desc_parts.append(region.text)

        description = " ".join(desc_parts).strip() or None
        if description is None and qty is None:
            continue  # a bare number row is not a line item

        unit_price = money_vals[0] if len(money_vals) >= 1 else None
        line_total = money_vals[-1] if len(money_vals) >= 2 else (
            money_vals[0] if qty in (None, 1) else None
        )
        needs_review = any(
            r.low_confidence_arabic or r.source == RecognizerSource.TROCR
            for r in row
        )

        items.append(
            LineItem(
                description=description,
                quantity=qty,
                unit_price=unit_price,
                line_total=line_total,
                needs_review=needs_review,
            )
        )
    return items
