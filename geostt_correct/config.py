from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Defaults tuned for weak STT + small local models."""

    ollama_host: str = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    ollama_model: str = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
    ollama_timeout_s: int = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
    temperature: float = float(os.environ.get("OLLAMA_TEMPERATURE", "0.0"))
    max_chunk_chars: int = int(os.environ.get("MAX_CHUNK_CHARS", "800"))
    sentence_overlap_chars: int = int(os.environ.get("SENTENCE_OVERLAP_CHARS", "0"))
    # Reject model output if too different from input (reduces hallucinated rewrites).
    min_sequence_ratio: float = float(os.environ.get("MIN_SEQUENCE_RATIO", "0.75"))
    max_relative_length: float = float(os.environ.get("MAX_RELATIVE_LENGTH", "1.30"))
    # Local Ollama pass; set GEOSTT_OLLAMA=0 to skip entirely (no-op pipeline).
    use_ollama: bool = True


def load_settings() -> Settings:
    ollama = os.environ.get("GEOSTT_OLLAMA", "1").strip() not in ("0", "false", "False")
    return Settings(use_ollama=ollama)
