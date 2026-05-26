from __future__ import annotations

import re
import unicodedata

# Georgian Mkhedruli + Mtavruli + common punctuation used in transcripts
_GEORGIAN_RE = re.compile(r"[\u10a0-\u10ff\u1c90-\u1cbf]")


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


def should_skip_llm(text: str) -> tuple[bool, str]:
    """
    If True, we keep the segment as-is (too broken or non-Georgian for a rewrite model).
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

    return False, "ok"
