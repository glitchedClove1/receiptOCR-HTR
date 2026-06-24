"""Tests for PDF support: rasterization, per-page processing, the
/v1/process-pdf endpoint, caching, and error handling.

PyMuPDF is required to build the test fixtures; if it isn't installed these
tests skip. OCR/HTR stages are mocked so no heavy models are needed.
"""
from __future__ import annotations

import importlib
from unittest import mock

import pytest
from fastapi.testclient import TestClient

import ocr_service.pipeline.orchestrator as orch
from ocr_service.pipeline.schemas import Region, Script

fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")


def _make_pdf(pages: list[list[str]]) -> bytes:
    doc = fitz.open()
    for lines in pages:
        page = doc.new_page(width=420, height=300)
        y = 50
        for ln in lines:
            page.insert_text((40, y), ln, fontsize=12)
            y += 30
    data = doc.tobytes()
    doc.close()
    return data


def _reg(t, x, y):
    return Region(text=t, confidence=0.95,
                  polygon=[(x, y), (x + 80, y), (x + 80, y + 20), (x, y + 20)],
                  script=Script.LATIN)


_PAGE_REGIONS = [
    [_reg("SUPERMARCHE LE BON PRIX", 40, 20), _reg("Total TTC", 40, 200),
     _reg("2,40 DT", 300, 200)],
    [_reg("BOULANGERIE CENTRALE", 40, 20), _reg("Total TTC", 40, 200),
     _reg("1,50 DT", 300, 200)],
]


def _fake_ocr_factory():
    state = {"i": 0}

    def fake_ocr(image, settings=None):
        r = _PAGE_REGIONS[state["i"] % len(_PAGE_REGIONS)]
        state["i"] += 1
        return r

    return fake_ocr


# --- rasterization ---------------------------------------------------------
def test_iter_pdf_pages_yields_images():
    from ocr_service.pipeline.pdf_render import count_pages, iter_pdf_pages
    data = _make_pdf([["Page one"], ["Page two"], ["Page three"]])
    assert count_pages(data) == 3
    pages = list(iter_pdf_pages(data))
    assert [n for n, _ in pages] == [1, 2, 3]
    for _, img in pages:
        assert img.ndim == 3 and img.shape[2] == 3  # BGR


def test_invalid_pdf_raises():
    from ocr_service.exceptions import InvalidPdfError
    from ocr_service.pipeline.pdf_render import iter_pdf_pages
    with pytest.raises(InvalidPdfError):
        list(iter_pdf_pages(b"not a pdf"))


def test_pdf_page_cap(monkeypatch):
    from ocr_service.pipeline.config import Settings
    from ocr_service.pipeline.pdf_render import iter_pdf_pages
    data = _make_pdf([["a"], ["b"], ["c"], ["d"], ["e"]])
    settings = Settings(pdf_max_pages=2)
    pages = list(iter_pdf_pages(data, settings))
    assert len(pages) == 2  # capped


# --- per-page processing ---------------------------------------------------
def test_process_pdf_bytes_one_doc_per_page():
    data = _make_pdf([["x"], ["y"]])
    with mock.patch.object(orch, "run_ocr", _fake_ocr_factory()), \
         mock.patch.object(orch, "apply_fallback", lambda img, r, s=None: r):
        results = list(orch.process_pdf_bytes(data))
    assert [n for n, _ in results] == [1, 2]
    assert results[0][1].page_number == 1
    assert results[0][1].vendor.value == "SUPERMARCHE LE BON PRIX"
    assert results[1][1].vendor.value == "BOULANGERIE CENTRALE"


# --- endpoint + caching ----------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OCR_API_KEY", "")
    from ocr_service.pipeline.config import get_settings
    get_settings.cache_clear()
    with mock.patch.object(orch, "run_ocr", _fake_ocr_factory()), \
         mock.patch.object(orch, "apply_fallback", lambda img, r, s=None: r):
        import ocr_service.app as appmod
        importlib.reload(appmod)
        yield TestClient(appmod.app)
    get_settings.cache_clear()


def test_process_pdf_endpoint(client):
    data = _make_pdf([["x"], ["y"]])
    r = client.post("/v1/process-pdf",
                    files={"file": ("doc.pdf", data, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["cache_hit"] is False
    assert body["page_count"] == 2
    assert len(body["pages"]) == 2
    assert body["pages"][0]["page_number"] == 1
    assert body["pages"][0]["excel_filename"].endswith("_p1.xlsx")
    # Each page's workbook is downloadable.
    name = body["pages"][0]["excel_filename"]
    d = client.get(f"/v1/download/{name}")
    assert d.status_code == 200


def test_process_pdf_cache_hit(client):
    data = _make_pdf([["x"], ["y"]])
    files = {"file": ("doc.pdf", data, "application/pdf")}
    first = client.post("/v1/process-pdf", files=files)
    assert first.json()["cache_hit"] is False
    second = client.post("/v1/process-pdf", files=files)
    assert second.json()["cache_hit"] is True
    assert second.json()["page_count"] == 2


def test_process_pdf_rejects_image(client):
    r = client.post("/v1/process-pdf",
                    files={"file": ("x.png", b"\x89PNG", "image/png")})
    assert r.status_code == 415


def test_process_pdf_rejects_empty(client):
    r = client.post("/v1/process-pdf",
                    files={"file": ("d.pdf", b"", "application/pdf")})
    assert r.status_code == 400
