from __future__ import annotations

import difflib
import re
import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _words(s: str) -> list[str]:
    return re.findall(r"\S+", _nfc(s))


# Without audio, both forms are often valid; do not let the LLM "pick" one over the STT transcript.
# Extend this set with other confusable sg/pl or case alternates your ASR toggles between.
_AMBIGUOUS_EQUIVALENT_PAIRS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"ბავშვს", "ბავშვებს"}),
    }
)


def _is_ambiguous_single_token_swap(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if len(wa) != len(wb) or not wa:
        return False
    diffs = [(x, y) for x, y in zip(wa, wb) if x != y]
    if len(diffs) != 1:
        return False
    x, y = diffs[0]
    return frozenset({x, y}) in _AMBIGUOUS_EQUIVALENT_PAIRS


_SENT_FINAL = frozenset(".!?…")


def _rstrip_sentence_punct(s: str) -> str:
    t = _nfc(s).rstrip()
    while t and t[-1] in _SENT_FINAL:
        t = t[:-1].rstrip()
    return t


def accept_correction(original: str, corrected: str, *, min_sequence_ratio: float, max_relative_length: float) -> tuple[bool, str]:
    """
    Return (accept, reason). Conservative: prefer keeping bad STT over invented text.
    """
    o = _nfc(original).strip()
    c = _nfc(corrected).strip()
    if not c:
        return False, "empty_output"

    if o == c:
        return True, "unchanged"

    # Compare without trailing sentence punctuation so "." does not distort similarity.
    o_cmp = _rstrip_sentence_punct(o)
    c_cmp = _rstrip_sentence_punct(c)
    if o_cmp == c_cmp and o != c:
        # Only punctuation / trivial trailing change (e.g. model added ".")
        return False, "trailing_punctuation_only"

    ratio = difflib.SequenceMatcher(a=o_cmp, b=c_cmp).ratio()
    if ratio < min_sequence_ratio:
        return False, f"sequence_ratio_low:{ratio:.3f}"

    if len(o) > 0 and len(c) > max_relative_length * len(o) + 8:
        return False, "output_much_longer"

    # Penalize big word-count jumps on already-long inputs (often means rewrite/hallucination)
    ow, cw = _words(o), _words(c)
    if len(ow) >= 6 and len(cw) > len(ow) + 4:
        return False, "many_extra_words"

    # Guard: if only one token changed, but new token is suspiciously longer,
    # prefer original (pre-LLM) token to avoid gibberish like "მოვა" -> "მოვიმოტ".
    if len(ow) == len(cw):
        diffs = [(a, b) for a, b in zip(ow, cw) if a != b]
        if len(diffs) == 1:
            a, b = diffs[0]
            if len(a) <= 5 and len(b) >= len(a) + 2:
                return False, "single_token_expansion"


    # Hard rollback guard: if too many aligned tokens changed, it is likely a rewrite.
    # We compare aligned tokens only (zip) to keep this deterministic and fast.
    if ow:
        aligned = min(len(ow), len(cw))
        changed = sum(1 for a, b in zip(ow, cw) if a != b)
        # Count extra unmatched tokens as changed too.
        changed += abs(len(ow) - len(cw))
        if aligned > 0 and (changed / max(len(ow), 1)) > 0.70:
            return False, "too_many_tokens_changed"
    

    if _is_ambiguous_single_token_swap(o_cmp, c_cmp):
        return False, "ambiguous_equivalent_morphology"

    return True, "ok"
