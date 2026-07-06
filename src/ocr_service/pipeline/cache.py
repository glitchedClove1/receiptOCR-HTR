"""Caching layer.

A minimal pluggable cache interface so the on-disk implementation can later be
swapped for Redis without touching the service code. Keyed on the SHA-256 of
the uploaded image bytes: identical uploads return the prior result and the
already-generated workbook.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Protocol

from .config import Settings, get_settings
from .schemas import Document


def content_hash(data: bytes) -> str:
    """SHA-256 of raw upload bytes — the cache key for an image or a PDF."""
    return hashlib.sha256(data).hexdigest()


# Backward-compatible alias (image uploads).
image_hash = content_hash


def page_key(pdf_hash: str, page_number: int) -> str:
    """Cache key for one page of a PDF: '<pdf_hash>_p<N>'."""
    return f"{pdf_hash}_p{page_number}"


class CacheBackend(Protocol):
    def get(self, key: str) -> Document | None: ...
    def put(self, key: str, doc: Document) -> None: ...
    def excel_path(self, key: str) -> str: ...
    def has_excel(self, key: str) -> bool: ...
    def get_page_count(self, pdf_hash: str) -> int | None: ...
    def put_page_count(self, pdf_hash: str, count: int) -> None: ...


class DiskCache:
    """On-disk cache: JSON for the Document, plus the generated .xlsx."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.dir = self.settings.cache_dir
        os.makedirs(self.dir, exist_ok=True)

    # --- internal paths -----------------------------------------------------
    def _json_path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.json")

    def excel_path(self, key: str) -> str:
        return os.path.join(self.dir, f"{key}.xlsx")

    def has_excel(self, key: str) -> bool:
        return os.path.exists(self.excel_path(key))

    # --- TTL handling -------------------------------------------------------
    def _expired(self, path: str) -> bool:
        ttl = self.settings.cache_ttl_seconds
        if ttl <= 0:
            return False
        return (time.time() - os.path.getmtime(path)) > ttl

    # --- API ----------------------------------------------------------------
    def get(self, key: str) -> Document | None:
        jp = self._json_path(key)
        if not os.path.exists(jp) or self._expired(jp):
            return None
        try:
            with open(jp, encoding="utf-8") as fh:
                return Document.model_validate(json.load(fh))
        except Exception:
            return None

    def put(self, key: str, doc: Document) -> None:
        # Atomic write: a crash mid-dump must not leave a truncated JSON that
        # later parses as a corrupt cache entry.
        final = self._json_path(key)
        tmp = f"{final}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc.model_dump(mode="json"), fh, ensure_ascii=False, indent=2)
        os.replace(tmp, final)

    # --- PDF page-count marker ----------------------------------------------
    # Without a stored page count, a full-cache check cannot know where a
    # PDF's pages end (walking until a miss always ends on a miss).
    def _pages_path(self, pdf_hash: str) -> str:
        return os.path.join(self.dir, f"{pdf_hash}_pages.json")

    def get_page_count(self, pdf_hash: str) -> int | None:
        pp = self._pages_path(pdf_hash)
        if not os.path.exists(pp) or self._expired(pp):
            return None
        try:
            with open(pp, encoding="utf-8") as fh:
                count = json.load(fh).get("page_count")
            return int(count) if count else None
        except Exception:
            return None

    def put_page_count(self, pdf_hash: str, count: int) -> None:
        pp = self._pages_path(pdf_hash)
        tmp = f"{pp}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"page_count": count}, fh)
        os.replace(tmp, pp)


def get_cache(settings: Settings | None = None) -> CacheBackend:
    """Factory — swap this to return a Redis-backed cache in future."""
    return DiskCache(settings)
