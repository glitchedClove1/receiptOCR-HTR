"""Unit tests for the pure-logic stages of the pipeline.

These run without PaddleOCR/TrOCR/torch installed: they cover preprocessing,
correction/number-parsing, script detection, the confidence-routing decision
(with a fake recognizer), spatial extraction, and Excel output. An end-to-end
test that exercises the real OCR models is marked and skipped unless a sample
image and the heavy deps are present.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from ocr_service.pipeline.config import Settings
from ocr_service.pipeline.correction import (
    correct_text,
    normalize_date,
    parse_currency,
    parse_number,
)
from ocr_service.pipeline.excel_writer import write_workbook
from ocr_service.pipeline.extraction import extract_document
from ocr_service.pipeline.htr_fallback import apply_fallback
from ocr_service.pipeline.preprocessing import preprocess
from ocr_service.pipeline.schemas import RecognizerSource, Region, Script
from ocr_service.pipeline.script_detect import detect_script


# ----------------------- helpers -------------------------------------------
def _reg(text, x, y, conf=0.95, script=Script.LATIN):
    return Region(
        text=text, confidence=conf,
        polygon=[(x, y), (x + 80, y), (x + 80, y + 20), (x, y + 20)],
        script=script,
    )


# ----------------------- correction / parsing ------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("1.234,56", 1234.56),
    ("1,234.56", 1234.56),
    ("1234,56", 1234.56),
    ("1 234,56", 1234.56),
    ("99.90", 99.9),
    ("€ 1.500,00", 1500.0),
    ("abc", None),
])
def test_parse_number(raw, expected):
    assert parse_number(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("12/03/2024", "2024-03-12"),
    ("2024-03-12", "2024-03-12"),
    ("12 mars 2024", "2024-03-12"),
    ("5 March 2023", "2023-03-05"),
    ("31-12-22", "2022-12-31"),
    ("no date here", None),
])
def test_normalize_date(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Total: 50 DT", "TND"),
    ("€100", "EUR"),
    ("USD 20", "USD"),
    ("nothing", None),
])
def test_parse_currency(raw, expected):
    assert parse_currency(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Total O0.5O", "Total 00.50"),   # numeric confusions fixed
    ("Solde", "Solde"),               # real word untouched
    ("Montant 1OO,5O", "Montant 100,50"),
    ("SOS", "SOS"),                   # all-confusable word, no digit -> untouched
])
def test_correct_text(raw, expected):
    assert correct_text(raw)[0] == expected


# ----------------------- script detection ----------------------------------
def test_detect_script():
    assert detect_script("Facture") == Script.LATIN
    assert detect_script("فاتورة") == Script.ARABIC
    assert detect_script("123.45") == Script.LATIN  # digits -> Latin


# ----------------------- preprocessing -------------------------------------
def test_preprocess_runs_and_records_ops():
    img = (np.random.rand(200, 300, 3) * 255).astype(np.uint8)
    out, applied = preprocess(img, Settings())
    assert out.ndim == 2  # grayscale single channel
    assert "grayscale" in applied
    assert "clahe" in applied


# ----------------------- confidence routing --------------------------------
def test_fallback_routes_low_conf_latin_to_recognizer():
    """A fake recognizer stands in for TrOCR to test the decision logic."""
    img = np.zeros((400, 600), dtype=np.uint8)
    regions = [
        _reg("good", 10, 10, conf=0.95),                       # kept
        _reg("bdd", 10, 50, conf=0.40),                        # -> fallback
        _reg("سعر", 10, 90, conf=0.30, script=Script.ARABIC),  # flagged only
    ]
    fake = {Script.LATIN: lambda crop: "FIXED"}
    out = apply_fallback(img, regions, Settings(), registry=fake)

    assert out[0].text == "good" and out[0].source == RecognizerSource.PADDLE
    assert out[1].text == "FIXED" and out[1].source == RecognizerSource.TROCR
    assert out[2].low_confidence_arabic is True
    assert out[2].source == RecognizerSource.PADDLE  # unchanged


# ----------------------- extraction ----------------------------------------
def test_extract_document_receipt():
    regions = [
        _reg("SUPERMARCHE LE BON PRIX", 40, 20),
        _reg("Facture N° F-2024-0087", 40, 50),
        _reg("Date: 12/03/2024", 40, 75),
        _reg("Lait demi-ecreme", 40, 130), _reg("2", 300, 130),
        _reg("1,20", 400, 130), _reg("2,40", 500, 130),
        _reg("Pain complet", 40, 160), _reg("1,50", 500, 160),
        _reg("Sous-total", 40, 220), _reg("3,90", 500, 220),
        _reg("TVA 7%", 40, 245), _reg("0,27", 500, 245),
        _reg("Total TTC", 40, 275), _reg("4,17 DT", 480, 275),
    ]
    doc = extract_document(regions)
    assert doc.vendor.value == "SUPERMARCHE LE BON PRIX"
    assert doc.invoice_number.value == "F-2024-0087"
    assert doc.document_date.value == "2024-03-12"
    assert doc.subtotal.value == "3.9"
    assert doc.tax.value == "0.27"
    assert doc.grand_total.value == "4.17"
    assert doc.currency.value == "TND"
    assert len(doc.line_items) == 2
    assert doc.line_items[0].quantity == 2
    assert doc.line_items[0].line_total == 2.4


# ----------------------- excel ---------------------------------------------
def test_write_workbook(tmp_path):
    regions = [_reg("ACME", 10, 10), _reg("Total TTC", 10, 80),
               _reg("12,00 EUR", 200, 80)]
    doc = extract_document(regions)
    path = write_workbook(doc, str(tmp_path / "out.xlsx"))
    assert os.path.exists(path)

    import openpyxl
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["Fields", "LineItems"]


# ----------------------- end-to-end (optional) -----------------------------
@pytest.mark.skipif(
    not os.path.exists("samples/sample_receipt.jpg"),
    reason="No sample image / heavy OCR deps not installed.",
)
def test_end_to_end_real_models():
    from ocr_service.pipeline.orchestrator import process_image_bytes
    with open("samples/sample_receipt.jpg", "rb") as fh:
        doc = process_image_bytes(fh.read())
    assert doc.region_count > 0
