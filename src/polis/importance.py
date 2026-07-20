"""Memory importance scoring — pluggable (Layer 1/2).

Two paths (ADR 0004): seeds carry an *authored* importance (no LLM cost), and an
optional LLM *poignancy* rater is available where the model's judgment earns its
cost. An ``ImportanceFn`` is ``(text) -> float`` in [1, 10]; ``Agent`` uses one
to score memories it writes back (R19).
"""
from __future__ import annotations

import re
from collections.abc import Callable

from . import prompts
from .llm import LLMClient

ImportanceFn = Callable[[str], float]

DEFAULT_SURVEY_IMPORTANCE = 5.0  # writeback default when no rater is configured


def constant(value: float = DEFAULT_SURVEY_IMPORTANCE) -> ImportanceFn:
    """Importance rater that ignores content and returns a fixed value."""

    def _fn(_text: str) -> float:
        return value

    return _fn


def llm_poignancy(client: LLMClient, *, default: float = DEFAULT_SURVEY_IMPORTANCE) -> ImportanceFn:
    """Rate poignancy 1-10 with a cheap LLM call; fall back to ``default`` on garbage."""

    def _fn(text: str) -> float:
        raw = client.complete(
            [{"role": "user", "content": prompts.poignancy(text)}],
            temperature=0.0,
            max_tokens=8,
        )
        m = re.search(r"\d+", raw or "")
        if not m:
            return default
        return float(min(10, max(1, int(m.group()))))

    return _fn
