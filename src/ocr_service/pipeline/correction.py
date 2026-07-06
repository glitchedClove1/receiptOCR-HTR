"""Post-processing / correction (no external service).

Two responsibilities:
  1. Fix OCR character confusions, but ONLY inside numeric contexts so we
     don't corrupt real words (e.g. don't turn 'Solde' into '5olde').
  2. Normalize numbers, currencies, and dates to canonical forms across the
     French and English conventions present in the documents.

Everything is local regex + small dictionaries; deterministic and testable.
"""
from __future__ import annotations

import re
from datetime import date

# Characters frequently misread by OCR inside digit runs -> their digit.
_NUMERIC_CONFUSIONS = str.maketrans(
    {"O": "0", "o": "0", "l": "1", "I": "1", "S": "5", "B": "8", "Z": "2"}
)

# Candidate tokens for numeric correction: runs of letters/digits/number
# punctuation. We only *apply* the confusable mapping to a candidate if it is
# already digit-dominant (see `_is_numeric_dominant`), so real words such as
# 'Solde' or 'Total' are never touched.
_CANDIDATE_TOKEN_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.,]*")


def _is_numeric_dominant(token: str) -> bool:
    """True if the token is dominated by digits (counting OCR confusables).

    The confusable letters (O,l,I,S,B,Z) are counted as *potential* digits,
    because inside a number they usually are. A token qualifies only if, after
    that mapping, true alphabetic characters are a minority. This corrects
    '1OO,5O' while still leaving 'Solde' (all real letters) and 'TVA' alone.
    """
    confusable = set("OolISBZ")
    digit_like = sum(1 for c in token if c.isdigit() or c in confusable)
    real_letters = sum(1 for c in token if c.isalpha() and c not in confusable)
    has_real_digit = any(c.isdigit() for c in token)
    return has_real_digit and digit_like > real_letters

_CURRENCY_SYMBOLS = {
    "€": "EUR", "$": "USD", "£": "GBP",
    "DT": "TND", "TND": "TND", "DH": "MAD",
    "EUR": "EUR", "USD": "USD", "GBP": "GBP",
}
_CURRENCY_RE = re.compile(
    r"(€|\$|£|\bDT\b|\bTND\b|\bDH\b|\bEUR\b|\bUSD\b|\bGBP\b)", re.IGNORECASE
)


def fix_numeric_token(token: str) -> str:
    """Apply confusable->digit mapping to a token assumed to be numeric."""
    return token.translate(_NUMERIC_CONFUSIONS)


def correct_text(text: str) -> tuple[str, bool]:
    """Correct OCR confusions in numeric tokens only.

    A token is corrected only if it is already digit-dominant, so words like
    'Solde' or 'Total' are never altered. Returns (corrected_text, changed).
    """
    changed = False

    def _repl(match: re.Match) -> str:
        nonlocal changed
        original = match.group(0)
        if not _is_numeric_dominant(original):
            return original
        candidate = original.translate(_NUMERIC_CONFUSIONS)
        if candidate != original:
            changed = True
        return candidate

    corrected = _CANDIDATE_TOKEN_RE.sub(_repl, text)
    return corrected, changed


def parse_number(raw: str) -> float | None:
    """Parse a numeric string in FR or EN convention to a float.

    Handles:
      '1.234,56' (FR)  -> 1234.56
      '1,234.56' (EN)  -> 1234.56
      '1234,56'        -> 1234.56
      '1 234,56'       -> 1234.56  (FR thin-space thousands)
      '99.90' / '99,90'-> 99.9
    Returns None if no number is present.
    """
    if raw is None:
        return None
    s = raw.strip()
    s, _ = correct_text(s)
    # Keep only digits, separators and sign.
    s = re.sub(r"[^0-9.,\-]", "", s)
    if not re.search(r"\d", s):
        return None

    has_comma = "," in s
    has_dot = "." in s

    if has_comma and has_dot:
        # The rightmost separator is the decimal mark.
        if s.rfind(",") > s.rfind("."):
            # FR: dot=thousands, comma=decimal
            s = s.replace(".", "").replace(",", ".")
        else:
            # EN: comma=thousands, dot=decimal
            s = s.replace(",", "")
    elif has_comma:
        # Comma alone: decimal if it looks like one (<=2 trailing digits), else
        # thousands.
        if re.search(r",\d{1,2}$", s):
            s = s.replace(",", ".")
        else:
            s = s.replace(",", "")
    # dot-only or plain integer needs no change.

    try:
        return float(s)
    except ValueError:
        return None


def parse_currency(text: str) -> str | None:
    """Return an ISO-ish currency code found in the text, if any."""
    m = _CURRENCY_RE.search(text)
    if not m:
        return None
    token = m.group(1).upper()
    return _CURRENCY_SYMBOLS.get(token) or _CURRENCY_SYMBOLS.get(m.group(1))


# --- Date normalization -----------------------------------------------------

_MONTHS = {
    # French
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
    "juin": 6, "juillet": 7, "août": 8, "aout": 8, "septembre": 9,
    "octobre": 10, "novembre": 11, "décembre": 12, "decembre": 12,
    # English
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}

_NUM_DATE_RE = re.compile(r"\b(\d{1,4})[/\-.](\d{1,2})[/\-.](\d{1,4})\b")
_TXT_DATE_RE = re.compile(
    r"\b(\d{1,2})\s+([A-Za-zÀ-ÿ]+)\.?\s+(\d{4})\b", re.UNICODE
)


def normalize_date(text: str) -> str | None:
    """Find and normalize the first date in `text` to ISO YYYY-MM-DD.

    Accepts numeric (dd/mm/yyyy, yyyy-mm-dd, dd-mm-yy) and textual
    ('12 mars 2024' / '12 March 2024') forms. Assumes day-first for ambiguous
    numeric dates, matching FR/EU receipts. Returns None if no valid date.
    """
    if not text:
        return None

    m = _TXT_DATE_RE.search(text)
    if m:
        day = int(m.group(1))
        month = _MONTHS.get(m.group(2).lower())
        year = int(m.group(3))
        if month:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # Try every numeric candidate, not just the first: a malformed reference
    # like '123/45/6789' earlier in the text must not mask a real date later.
    for m in _NUM_DATE_RE.finditer(text):
        a, b, c = m.group(1), m.group(2), m.group(3)
        if len(a) == 4:  # yyyy-mm-dd
            year, month, day = int(a), int(b), int(c)
        else:            # dd-mm-yyyy / dd-mm-yy (day-first)
            day, month = int(a), int(b)
            year = int(c)
            if year < 100:  # 2-digit year -> assume 2000s
                year += 2000
        try:
            return date(year, month, day).isoformat()
        except ValueError:
            continue

    return None
