"""Excel output (openpyxl) — one workbook per document.

Produces two sheets:
  * 'Fields'    — header key/value pairs + totals, with provenance columns
                  and a review flag.
  * 'LineItems' — the tabular line items.
Low-confidence / Arabic-flagged values are highlighted so a human reviewer can
find them quickly.
"""
from __future__ import annotations

import os

from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

from .config import Settings, get_settings
from .schemas import Document, FieldValue

_HEADER_FILL = PatternFill("solid", fgColor="305496")
_HEADER_FONT = Font(bold=True, color="FFFFFF")
_REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC")  # amber = needs review


def _row_for(label: str, fv: FieldValue) -> list:
    return [
        label,
        fv.value,
        fv.source.value if fv.source else None,
        round(fv.confidence, 4) if fv.confidence is not None else None,
        "yes" if fv.corrected else "no",
        "REVIEW" if fv.needs_review else "",
    ]


def write_workbook(
    doc: Document, path: str, settings: Settings | None = None
) -> str:
    """Write `doc` to an .xlsx workbook at `path`. Returns the path."""
    settings = settings or get_settings()
    wb = Workbook()

    # --- Fields sheet -------------------------------------------------------
    ws = wb.active
    ws.title = "Fields"
    headers = ["Field", "Value", "Source", "Confidence", "Corrected", "Review"]
    ws.append(headers)
    for col, _ in enumerate(headers, start=1):
        c = ws.cell(row=1, column=col)
        c.fill = _HEADER_FILL
        c.font = _HEADER_FONT

    field_rows = [
        ("Vendor", doc.vendor),
        ("Document Date", doc.document_date),
        ("Invoice Number", doc.invoice_number),
        ("Vendor Tax ID", doc.vendor_tax_id),
        ("Subtotal", doc.subtotal),
        ("Tax / TVA", doc.tax),
        ("Grand Total", doc.grand_total),
        ("Currency", doc.currency),
    ]
    for label, fv in field_rows:
        row = _row_for(label, fv)
        ws.append(row)
        if fv.needs_review:
            for col in range(1, len(headers) + 1):
                ws.cell(row=ws.max_row, column=col).fill = _REVIEW_FILL

    # Diagnostics block
    ws.append([])
    ws.append(["Regions detected", doc.region_count])
    ws.append(["Low-confidence regions", doc.low_confidence_count])
    ws.append(["Preprocessing", ", ".join(doc.preprocessing_applied)])

    _autosize(ws, headers)

    # --- LineItems sheet ----------------------------------------------------
    ws2 = wb.create_sheet("LineItems")
    li_headers = ["Description", "Quantity", "Unit Price", "Line Total", "Review"]
    ws2.append(li_headers)
    for col, _ in enumerate(li_headers, start=1):
        c = ws2.cell(row=1, column=col)
        c.fill = _HEADER_FILL
        c.font = _HEADER_FONT

    for li in doc.line_items:
        ws2.append([
            li.description, li.quantity, li.unit_price, li.line_total,
            "REVIEW" if li.needs_review else "",
        ])
        if li.needs_review:
            for col in range(1, len(li_headers) + 1):
                ws2.cell(row=ws2.max_row, column=col).fill = _REVIEW_FILL

    _autosize(ws2, li_headers)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    wb.save(path)
    return path


def _autosize(ws, headers) -> None:
    """Roughly fit column widths to content."""
    for col_idx in range(1, len(headers) + 1):
        letter = get_column_letter(col_idx)
        longest = max(
            (len(str(ws.cell(row=r, column=col_idx).value or ""))
             for r in range(1, ws.max_row + 1)),
            default=10,
        )
        ws.column_dimensions[letter].width = min(max(longest + 2, 12), 48)
