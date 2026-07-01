"""Pydantic schemas shared across the pipeline.

These define the contract between stages: the OCR engine emits `Region`s,
the router/correction stages enrich them, and extraction produces a
`Document`. Keeping them in one place makes each stage independently
testable against a stable shape.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# A polygon is a list of (x, y) points (PaddleOCR returns 4-point quads).
Polygon = list[tuple[float, float]]


class Script(str, Enum):
    LATIN = "latin"
    ARABIC = "arabic"
    OTHER = "other"


class RecognizerSource(str, Enum):
    PADDLE = "paddle"
    TROCR = "trocr"


class Region(BaseModel):
    """A single detected + recognized text region."""

    text: str
    confidence: float
    polygon: Polygon
    script: Script = Script.LATIN
    source: RecognizerSource = RecognizerSource.PADDLE

    # Populated when the HTR fallback runs on this region.
    paddle_text: str | None = None
    paddle_confidence: float | None = None
    trocr_text: str | None = None
    # True when Paddle conf was low but region was Arabic so TrOCR was skipped.
    low_confidence_arabic: bool = False
    corrected: bool = False

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box (x_min, y_min, x_max, y_max)."""
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return (min(xs), min(ys), max(xs), max(ys))

    @property
    def center_y(self) -> float:
        ys = [p[1] for p in self.polygon]
        return sum(ys) / len(ys)

    @property
    def center_x(self) -> float:
        xs = [p[0] for p in self.polygon]
        return sum(xs) / len(xs)


class FieldValue(BaseModel):
    """An extracted field with provenance for human review."""

    value: str | None = None
    source: RecognizerSource | None = None
    confidence: float | None = None
    corrected: bool = False
    needs_review: bool = False


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    line_total: float | None = None
    needs_review: bool = False


class Document(BaseModel):
    """Final structured output for one processed document."""

    vendor: FieldValue = Field(default_factory=FieldValue)
    document_date: FieldValue = Field(default_factory=FieldValue)
    invoice_number: FieldValue = Field(default_factory=FieldValue)
    vendor_tax_id: FieldValue = Field(default_factory=FieldValue)

    line_items: list[LineItem] = Field(default_factory=list)

    subtotal: FieldValue = Field(default_factory=FieldValue)
    tax: FieldValue = Field(default_factory=FieldValue)
    grand_total: FieldValue = Field(default_factory=FieldValue)
    currency: FieldValue = Field(default_factory=FieldValue)

    # Dynamic fields extracted by the LLM enrichment pass (anything beyond the
    # fixed schema above that appears on the document).
    extra_fields: dict[str, "FieldValue"] = Field(default_factory=dict)

    # Diagnostics
    preprocessing_applied: list[str] = Field(default_factory=list)
    region_count: int = 0
    low_confidence_count: int = 0
    # Set when the document originates from a PDF page (1-indexed); None for
    # single-image uploads.
    page_number: int | None = None


class ProcessResponse(BaseModel):
    cache_hit: bool
    image_hash: str
    excel_filename: str
    document: Document


class PageResult(BaseModel):
    """One processed PDF page: its document plus its own workbook handle."""

    page_number: int
    excel_filename: str
    document: Document


class PdfProcessResponse(BaseModel):
    """Result of processing a PDF: one entry per page, each with a workbook."""

    cache_hit: bool
    pdf_hash: str
    page_count: int
    pages: list[PageResult]
