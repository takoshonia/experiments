from __future__ import annotations

import re
import unicodedata


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


# Split on sentence-like boundaries (Georgian + Latin common marks)
_SPLIT_RE = re.compile(r"(?<=[\.\!\?\u0589\u061f])\s+")


def chunk_by_sentences(text: str, max_chunk_chars: int) -> list[str]:
    """
    Merge sentences into chunks under max_chunk_chars to limit latency/RAM on small models.
    """
    t = _nfc(text).strip()
    if not t:
        return []

    parts = [p.strip() for p in _SPLIT_RE.split(t) if p.strip()]
    if not parts:
        return [t] if len(t) <= max_chunk_chars else _hard_split(t, max_chunk_chars)

    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in parts:
        if not buf:
            buf.append(p)
            buf_len = len(p)
            continue
        if buf_len + 1 + len(p) <= max_chunk_chars:
            buf.append(p)
            buf_len += 1 + len(p)
        else:
            chunks.append(" ".join(buf))
            buf = [p]
            buf_len = len(p)
    if buf:
        chunks.append(" ".join(buf))

    # If a single "sentence" is huge, hard-split
    out: list[str] = []
    for ch in chunks:
        if len(ch) <= max_chunk_chars:
            out.append(ch)
        else:
            out.extend(_hard_split(ch, max_chunk_chars))
    return out


def _hard_split(text: str, max_chunk_chars: int) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + max_chunk_chars].strip())
        i += max_chunk_chars
    return [x for x in out if x]
