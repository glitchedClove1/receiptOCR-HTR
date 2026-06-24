"""Authentication (stub).

A minimal API-key check wired in as a FastAPI dependency. It is intentionally
simple — a placeholder to be replaced with the platform's real auth (JWT,
mTLS, gateway-injected identity, etc.). When `api_key` is empty in settings,
auth is disabled so the service can run behind a trusted gateway unchanged.

To harden later: swap the constant-time string compare for a real token
validator, add scopes/roles, and rate-limit per principal.
"""
from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from .pipeline.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the X-API-Key header when configured."""
    settings = get_settings()
    if not settings.api_key:
        return  # auth disabled — handled upstream
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
