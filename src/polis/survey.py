"""Survey question + answer schemas (Layer 5 — Interface/Query)."""
from __future__ import annotations

from pydantic import BaseModel, Field


class SurveyQuestion(BaseModel):
    """A single-select survey question (R18: a separately invokable query)."""

    text: str
    options: list[str] = Field(min_length=2)


class SurveyAnswer(BaseModel):
    """One agent's single-select answer with a short rationale."""

    choice: str
    reason: str
