"""Script detection.

A lightweight, dependency-free heuristic that classifies a string as Latin,
Arabic, or Other based on Unicode ranges. Used by the recognizer router to
decide whether the TrOCR (Latin-only) fallback applies.
"""
from __future__ import annotations

from .schemas import Script

# Arabic Unicode blocks: Arabic (0600-06FF), Supplement (0750-077F),
# Extended-A (08A0-08FF), Presentation Forms-A (FB50-FDFF), Forms-B (FE70-FEFF).
_ARABIC_RANGES = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)


def _is_arabic_char(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _ARABIC_RANGES)


def _is_latin_char(ch: str) -> bool:
    cp = ord(ch)
    return (0x0041 <= cp <= 0x024F)  # basic + extended Latin


def detect_script(text: str) -> Script:
    """Classify by majority of *alphabetic* characters; ties favour Latin."""
    arabic = latin = 0
    for ch in text:
        if _is_arabic_char(ch):
            arabic += 1
        elif _is_latin_char(ch):
            latin += 1
    if arabic == 0 and latin == 0:
        # Pure digits/punctuation — treat as Latin so numeric regions can
        # still benefit from the TrOCR fallback.
        return Script.LATIN
    if arabic > latin:
        return Script.ARABIC
    return Script.LATIN
