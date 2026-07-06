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
# Cues that unambiguously mean the grand total. A row hitting one of these is
# never routed to tax, even if it also contains "taxes" (e.g. the standard FR
# boilerplate "TOTAL TTC toutes taxes comprises").
_GRAND_STRONG_CUES = ("total ttc", "grand total", "total a payer",
                      "net a payer", "montant total")
# NB: bare "ht" is intentionally NOT a cue — it substring-matches English
# words (freight, light, weight...). " ht " (word-bounded) covers "Montant HT".
_SUBTOTAL_CUES = ("sous-total", "sous total", "subtotal", "total ht", " ht ")
_TAX_CUES = ("tva", "vat", "tax", "taxe")
_INVOICE_CUES = ("facture", "invoice", "recu", "reçu", "ticket", "n°", "no.", "no ")
_TAXID_CUES = ("matricule fiscal", "tax id", "vat number", "siret", "if ",
               "mf ", "rc ")
_DATE_CUES = ("date", "le ")

_QTY_RE = re.compile(r"^\s*(\d{1,3})\s*[xX]?\s*$")
# Matches a full monetary token, including grouped thousands ("1.234,56",
# "1,234.56", "1 234,56") and 2-3 decimal places (TND uses 3). The grouped
# alternative must come first so finditer consumes the whole token instead of
# fragmenting it (e.g. "1.234,56" must not match as "1.23" + "4,56").
_MONEY_RE = re.compile(r"\d{1,3}(?:[ \u00A0.,]\d{3})+[.,]\d{2,3}|\d+[.,]\d{2,3}")


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


def _needs_review(region: Region) -> bool:
    """A region needs human review if it is low-confidence Arabic, came from
    the HTR fallback, or the fallback was attempted but failed."""
    return (region.low_confidence_arabic
            or region.source == RecognizerSource.TROCR
            or region.fallback_failed)


def _field_from_region(region: Region, value: str, corrected: bool) -> FieldValue:
    """Build a FieldValue, flagging review for unreliable regions."""
    return FieldValue(
        value=value,
        source=region.source,
        confidence=region.confidence,
        corrected=corrected,
        needs_review=_needs_review(region),
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
    doc.low_confidence_count = sum(1 for r in regions if _needs_review(r))

    rows = _group_rows(regions)

    # --- Vendor: heuristic = first non-empty row near the top, not a cue. ----
    for row in rows[:3]:
        joined = " ".join(r.text for r in row).strip()
        if joined and not _has_cue(joined.lower(), _TOTAL_CUES + _TAX_CUES):
            doc.vendor = _field_from_region(row[0], joined, False)
            break

    currency_seen: str | None = None
    # First date seen WITHOUT a "Date:" cue — used only if no cued date exists,
    # so a billing-period or delivery date can't hijack the document date.
    uncued_date: tuple[str, Region] | None = None

    # --- Scan rows for header fields and totals ------------------------------
    for row in rows:
        joined = " ".join(r.text for r in row)
        low = joined.lower()

        if currency_seen is None:
            cur = parse_currency(joined)
            if cur:
                currency_seen = cur

        # Date — cued rows win; un-cued dates are only a post-loop fallback.
        if doc.document_date.value is None and _has_cue(low, _DATE_CUES):
            iso = normalize_date(joined)
            if iso:
                doc.document_date = _field_from_region(row[0], iso, True)
        elif uncued_date is None:
            iso = normalize_date(joined)
            if iso:
                uncued_date = (iso, row[0])

        # Invoice number. NB: no confusable correction here — identifiers like
        # 'IN2024-001' are not numeric context and correction corrupts them.
        if doc.invoice_number.value is None and _has_cue(low, _INVOICE_CUES):
            m = re.search(r"(?:n[°o]\.?|no\.?|#)\s*([A-Za-z0-9\-/]+)", joined,
                          re.IGNORECASE)
            if m:
                doc.invoice_number = _field_from_region(row[0], m.group(1), False)

        # Tax ID — must contain at least one digit so an uppercase label
        # ("MATRICULE FISCAL ...") is never captured as the ID itself.
        if doc.vendor_tax_id.value is None and _has_cue(low, _TAXID_CUES):
            m = re.search(r"(?=[A-Z0-9/\-]*\d)([A-Z0-9]{6,}[A-Z0-9/\-]*)", joined)
            if m:
                doc.vendor_tax_id = _field_from_region(row[0], m.group(1), False)

        # Totals routing. A row with a strong grand-total cue ("TOTAL TTC ...")
        # goes to grand_total even when it also mentions taxes ("toutes taxes
        # comprises"); otherwise tax cues win, then subtotal, then bare total.
        money = _extract_money_after_cue(row)
        if money:
            val, region = money
            strong_total = _has_cue(low, _GRAND_STRONG_CUES)
            if (_has_cue(low, _TAX_CUES) and not strong_total
                    and doc.tax.value is None):
                doc.tax = _field_from_region(region, str(val), True)
            elif (_has_cue(low, _SUBTOTAL_CUES) and not strong_total
                    and doc.subtotal.value is None):
                doc.subtotal = _field_from_region(region, str(val), True)
            elif _has_cue(low, _TOTAL_CUES) and doc.grand_total.value is None:
                doc.grand_total = _field_from_region(region, str(val), True)

    if doc.document_date.value is None and uncued_date is not None:
        doc.document_date = _field_from_region(uncued_date[1], uncued_date[0], True)

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

        # Description = leftmost text before the first money token. When OCR
        # merges description and price into one region ("Cafe Express 3,50"),
        # the text before the money token inside that region is still used —
        # the row must not be dropped. A standalone small integer is taken as
        # quantity regardless of position (receipt layouts vary).
        desc_parts: list[str] = []
        qty: float | None = None
        for region in row:
            money_match = _MONEY_RE.search(region.text)
            if money_match:
                prefix = region.text[:money_match.start()].strip()
                if prefix:
                    qm = _QTY_RE.match(prefix)
                    if qm and qty is None:
                        qty = float(qm.group(1))
                    else:
                        desc_parts.append(prefix)
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
