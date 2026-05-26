from __future__ import annotations

from dataclasses import dataclass, field

from geostt_correct.chunking import chunk_by_sentences
from geostt_correct.config import Settings, load_settings
from geostt_correct.gating import should_skip_llm
from geostt_correct.ollama_backend import correct_chunk
from geostt_correct.safety import accept_correction


@dataclass
class SegmentResult:
    source: str
    output: str
    skipped_llm: bool
    skip_reason: str = ""
    rejected_model: bool = False
    reject_reason: str = ""
    pre_llm: str | None = None


@dataclass
class DocumentResult:
    text: str
    segments: list[SegmentResult] = field(default_factory=list)


def correct_document(text: str, settings: Settings | None = None) -> DocumentResult:
    settings = settings or load_settings()
    chunks = chunk_by_sentences(text, settings.max_chunk_chars)
    if not chunks:
        return DocumentResult(text="", segments=[])

    segments: list[SegmentResult] = []
    outs: list[str] = []

    for ch in chunks:
        raw = ch
        working = raw

        if not settings.use_ollama:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=True,
                    skip_reason="ollama_disabled",
                )
            )
            outs.append(working)
            continue

        skip, reason = should_skip_llm(working)
        if skip:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=True,
                    skip_reason=reason,
                )
            )
            outs.append(working)
            continue

        try:
            cand = correct_chunk(
                working,
                host=settings.ollama_host,
                model=settings.ollama_model,
                temperature=settings.temperature,
                timeout_s=settings.ollama_timeout_s,
            )
        except Exception:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=False,
                    rejected_model=True,
                    reject_reason="ollama_error",
                )
            )
            outs.append(working)
            continue

        ok, why = accept_correction(
            working,
            cand,
            min_sequence_ratio=settings.min_sequence_ratio,
            max_relative_length=settings.max_relative_length,
        )
        if not ok:
            segments.append(
                SegmentResult(
                    source=raw,
                    output=working,
                    skipped_llm=False,
                    rejected_model=True,
                    reject_reason=why,
                )
            )
            outs.append(working)
            continue

        segments.append(SegmentResult(source=raw, output=cand, skipped_llm=False))
        outs.append(cand)

    return DocumentResult(text=" ".join(outs).strip(), segments=segments)
