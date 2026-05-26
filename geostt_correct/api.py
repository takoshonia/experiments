from __future__ import annotations

from dataclasses import replace

from fastapi import FastAPI
from pydantic import BaseModel

from geostt_correct.config import load_settings
from geostt_correct.pipeline import correct_document


class CorrectRequest(BaseModel):
    text: str = ""


class CorrectResponse(BaseModel):
    text: str
    llm_applied: bool | None = None
    llm_attempted: bool | None = None


app = FastAPI(title="geostt-correct API", version="0.1.0")


@app.post("/v1/correct/prellm", response_model=CorrectResponse)
def correct_prellm(req: CorrectRequest) -> CorrectResponse:
    settings = load_settings()
    pre_settings = replace(settings, use_ollama=False)
    result = correct_document(req.text, pre_settings)
    return CorrectResponse(
        text=result.text,
        llm_applied=False,
        llm_attempted=False,
    )


@app.post("/v1/correct/final", response_model=CorrectResponse)
def correct_final(req: CorrectRequest) -> CorrectResponse:
    settings = load_settings()
    result = correct_document(req.text, settings)
    llm_attempted = any(not s.skipped_llm for s in result.segments)
    llm_applied = any(
        (not s.skipped_llm)
        and (
            s.output
            != (s.pre_llm if s.pre_llm is not None else s.source)
        )
        for s in result.segments
    )
    return CorrectResponse(
        text=result.text,
        llm_applied=llm_applied,
        llm_attempted=llm_attempted,
    )
