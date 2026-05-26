from __future__ import annotations

import os
import re
import unicodedata

# Georgian Mkhedruli + Mtavruli + common punctuation used in transcripts
_GEORGIAN_RE = re.compile(r"[\u10a0-\u10ff\u1c90-\u1cbf]")
_REPEATED_CHAR_RE = re.compile(r"(.)\1{3,}")  # 4+ same char in a row -> likely ASR glitch
_LATIN_CYRILLIC_RE = re.compile(r"[A-Za-z\u0400-\u04ff]")  # mixed-script -> likely ASR glitch


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def georgian_letter_ratio(text: str) -> float:
    t = _nfc(text).strip()
    if not t:
        return 0.0
    letters = sum(1 for ch in t if ch.isalpha())
    if letters == 0:
        return 0.0
    geo = sum(1 for ch in t if _GEORGIAN_RE.match(ch))
    return geo / letters


def word_count(text: str) -> int:
    t = _nfc(text).strip()
    if not t:
        return 0
    return len(re.findall(r"\S+", t))


def looks_clean(text: str) -> tuple[bool, str]:
    """Return (True, reason) only if the input looks already clean (no ASR-failure markers).

    Conservative on purpose: it is fine to let some clean rows through (false negative),
    we only want to avoid mis-flagging a *broken* row as clean (false positive).

    Markers that disqualify "clean":
      - 4+ repeated chars in a row (ASR stutter / decode loop)
      - any Latin or Cyrillic letter (script mixing -> usually wrong)
      - any word longer than 20 characters (run-on / joined words)
      - Georgian letter ratio below threshold (default 0.97; tune with GEOSTT_CLEAN_RATIO)
    """
    t = _nfc(text).strip()
    if not t:
        return False, "empty"
    if _REPEATED_CHAR_RE.search(t):
        return False, "repeated_char_run"
    if _LATIN_CYRILLIC_RE.search(t):
        return False, "mixed_script"
    words = re.findall(r"\S+", t)
    if any(len(w) > 20 for w in words):
        return False, "very_long_word"
    threshold = float(os.environ.get("GEOSTT_CLEAN_RATIO", "0.97"))
    ratio = georgian_letter_ratio(t)
    if ratio < threshold:
        return False, f"georgian_ratio_low:{ratio:.2f}"
    return True, "looks_clean"


def should_skip_llm(text: str) -> tuple[bool, str]:
    """
    If True, we keep the segment as-is (too broken or non-Georgian for a rewrite model,
    OR already clean enough that touching it is pure downside).
    This avoids turning garbage into confident hallucinations.
    """
    t = _nfc(text).strip()
    if not t:
        return True, "empty"

    wc = word_count(t)
    if wc < 2:
        return True, "too_few_words"

    ratio = georgian_letter_ratio(t)
    if ratio < 0.55:
        return True, "low_georgian_letter_ratio"

    # Very short "sentence" with odd tokens (heuristic for collapsed STT like "დედა ლრვ")
    if wc == 2 and len(t) < 12:
        return True, "very_short_two_token"

    # Skip when the input already looks clean. Disable with GEOSTT_SKIP_CLEAN=0.
    if os.environ.get("GEOSTT_SKIP_CLEAN", "1").strip() not in ("0", "false", "False"):
        clean, why = looks_clean(t)
        if clean:
            return True, f"looks_clean:{why}"

    return False, "ok"
