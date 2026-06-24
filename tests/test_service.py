"""Tests for the upgraded service layer: auth, error mapping, request id,
and the thread-safe model loader's locking behaviour.

These mock the OCR/HTR stages so no heavy models are required.
"""
from __future__ import annotations

import threading
from unittest import mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import ocr_service.pipeline.orchestrator as orch
from ocr_service.exceptions import ModelLoadError
from ocr_service.pipeline.schemas import Region, Script


def _png_bytes() -> bytes:
    img = (np.ones((200, 320, 3)) * 255).astype("uint8")
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


def _fake_regions(image, settings=None):
    def reg(t, x, y):
        return Region(text=t, confidence=0.95,
                      polygon=[(x, y), (x + 60, y), (x + 60, y + 18), (x, y + 18)],
                      script=Script.LATIN)
    return [reg("ACME", 20, 20), reg("Total TTC", 20, 120), reg("9,99 EUR", 200, 120)]


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Isolate cache to a temp dir and stub the OCR stages.
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("OCR_API_KEY", "")  # auth disabled by default
    from ocr_service.pipeline.config import get_settings
    get_settings.cache_clear()  # re-read env

    with mock.patch.object(orch, "run_ocr", _fake_regions), \
         mock.patch.object(orch, "apply_fallback", lambda img, r, s=None: r):
        import importlib

        import ocr_service.app as appmod
        importlib.reload(appmod)
        yield TestClient(appmod.app)
    get_settings.cache_clear()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_process_and_cache(client):
    data = _png_bytes()
    r1 = client.post("/v1/process", files={"file": ("r.png", data, "image/png")})
    assert r1.status_code == 200
    body = r1.json()
    assert body["cache_hit"] is False
    assert body["document"]["grand_total"]["value"] == "9.99"
    # Correlation id echoed back.
    assert "X-Request-ID" in r1.headers

    r2 = client.post("/v1/process", files={"file": ("r.png", data, "image/png")})
    assert r2.json()["cache_hit"] is True


def test_unsupported_media_type(client):
    r = client.post("/v1/process", files={"file": ("x.txt", b"hi", "text/plain")})
    assert r.status_code == 415


def test_empty_upload(client):
    r = client.post("/v1/process", files={"file": ("r.png", b"", "image/png")})
    assert r.status_code == 400


def test_invalid_image_bytes(client):
    r = client.post("/v1/process",
                    files={"file": ("r.png", b"not-an-image", "image/png")})
    assert r.status_code == 400


def test_model_load_error_maps_to_503(client):
    def boom(image, settings=None):
        raise ModelLoadError("paddle missing")
    with mock.patch.object(orch, "run_ocr", boom):
        r = client.post("/v1/process",
                        files={"file": ("r.png", _png_bytes(), "image/png")})
    assert r.status_code == 503


def test_download_path_traversal_blocked(client):
    r = client.get("/v1/download/..%2f..%2fetc%2fpasswd")
    assert r.status_code in (400, 404)


def test_auth_enforced_when_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_CACHE_DIR", str(tmp_path / "c"))
    monkeypatch.setenv("OCR_API_KEY", "secret123")
    from ocr_service.pipeline.config import get_settings
    get_settings.cache_clear()
    with mock.patch.object(orch, "run_ocr", _fake_regions), \
         mock.patch.object(orch, "apply_fallback", lambda img, r, s=None: r):
        import importlib

        import ocr_service.app as appmod
        importlib.reload(appmod)
        c = TestClient(appmod.app)
        data = _png_bytes()
        # No key -> 401
        assert c.post("/v1/process",
                      files={"file": ("r.png", data, "image/png")}).status_code == 401
        # Correct key -> 200
        ok = c.post("/v1/process", files={"file": ("r.png", data, "image/png")},
                    headers={"X-API-Key": "secret123"})
        assert ok.status_code == 200
    get_settings.cache_clear()


def test_model_loader_constructs_once_under_threads(monkeypatch):
    """The thread-safe loader must build the model exactly once."""
    from ocr_service.pipeline import model_loader
    model_loader.reset_models()
    calls = {"n": 0}

    class _FakePaddle:
        def __init__(self, **kw):
            calls["n"] += 1

    fake_module = mock.MagicMock()
    fake_module.PaddleOCR = _FakePaddle

    from ocr_service.pipeline.config import Settings
    settings = Settings()

    with mock.patch.dict("sys.modules", {"paddleocr": fake_module}):
        results = []

        def worker():
            results.append(model_loader.get_paddle(settings))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    assert calls["n"] == 1                 # constructed once
    assert len({id(r) for r in results}) == 1  # all got the same instance
    model_loader.reset_models()
