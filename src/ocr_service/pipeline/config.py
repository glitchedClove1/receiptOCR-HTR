"""Central configuration for the OCR service.

All tunable values live here and are overridable via environment variables
or a `.env` file. Nothing in the pipeline should hard-code a threshold,
path, or model name — it should read it from this settings object.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, populated from environment / .env."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="OCR_", extra="ignore"
    )

    # --- Confidence routing -------------------------------------------------
    # PaddleOCR regions scoring below this trigger the TrOCR handwriting pass.
    confidence_threshold: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Below this Paddle confidence, route Latin regions to TrOCR.",
    )

    # --- Languages ----------------------------------------------------------
    # PaddleOCR language packs to enable. 'fr','en' cover Latin; 'ar' handles
    # PRINTED Arabic. Handwritten Arabic is out of scope in this version.
    enabled_languages: list[str] = Field(default_factory=lambda: ["fr", "en", "ar"])
    # Primary PaddleOCR lang model used for the first detection+recognition pass.
    paddle_primary_lang: str = Field(default="fr")

    # --- Models -------------------------------------------------------------
    trocr_model_name: str = Field(default="microsoft/trocr-large-handwritten")
    # Swap to 'microsoft/trocr-base-handwritten' on CPU-only/low-memory hosts.

    # --- Pre-processing toggles --------------------------------------------
    pre_grayscale: bool = True
    pre_clahe: bool = True
    pre_denoise: bool = True
    pre_deskew: bool = True

    # --- Caching ------------------------------------------------------------
    cache_dir: str = Field(default="./.ocr_cache")
    cache_ttl_seconds: int = Field(default=60 * 60 * 24 * 7)  # 7 days; 0 = no expiry
    output_dir: str = Field(default="./outputs")

    # --- Service ------------------------------------------------------------
    max_upload_mb: int = Field(default=25)
    log_level: str = Field(default="INFO")

    # --- PDF handling -------------------------------------------------------
    # DPI to rasterize each PDF page before OCR. 200-300 is a good range:
    # higher is sharper but slower and heavier on memory.
    pdf_render_dpi: int = Field(default=220, ge=72, le=600)
    # Safety cap on pages processed from one PDF (protects against huge files).
    pdf_max_pages: int = Field(default=50, ge=1)

    # --- Auth (stub) --------------------------------------------------------
    # When set, /process and /download require header `X-API-Key: <value>`.
    # Leave empty to disable (e.g. when auth is handled at the gateway).
    api_key: str = Field(default="")

    # --- Device -------------------------------------------------------------
    use_gpu: bool = Field(default=False)

    # --- LLM enrichment (Ollama) --------------------------------------------
    # Set OCR_OLLAMA_ENABLED=false to skip the LLM pass entirely.
    ollama_enabled: bool = Field(default=True)
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="qwen2.5:3b")
    ollama_timeout_seconds: int = Field(default=180)


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor so the object is constructed once."""
    return Settings()
