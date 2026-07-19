"""OpenRouter-backed LLM client (Layer 1 — Engine).

Phases 0–2 use a managed per-token provider (OpenRouter); the model is pinned
and logged per call (R6) and generation params are per-call/per-agent (R1).
Single-select output is schema-guided via ``response_format`` json_schema — the
managed-tier stand-in for the grammar-constrained decoding (R20) we get from a
self-hosted vLLM at Phase 5+. Rationale: docs/adr/0002-phased-llm-provider.md.

Qwen3 runs in "thinking" mode by default, which burns tokens and (with a JSON
schema) can derail the output; we disable it for survey answers.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "qwen/qwen3-8b"


class LLMError(RuntimeError):
    """Raised when the endpoint fails to return a usable, in-vocabulary answer."""


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.8  # per-agent generation param (R1)
    top_p: float = 1.0
    max_tokens: int = 512
    reasoning: bool = False  # Qwen3 thinking off for survey answers


class LLMClient:
    """Thin OpenAI-compatible client pointed at OpenRouter."""

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None):
        self.config = config or LLMConfig()
        load_dotenv(find_dotenv(usecwd=True))
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set (add it to .env).")
        self._client = OpenAI(base_url=self.config.base_url, api_key=key)

    def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            extra_body={"reasoning": {"enabled": self.config.reasoning}},
        )
        return resp.choices[0].message.content

    def choose(self, *, system, user, options, temperature=None, max_tokens=300, retries=1) -> dict:
        """Single-select with a short rationale, validated against ``options``.

        Returns ``{"choice", "reason", "model", "usage"}``. Raises ``LLMError``
        if no schema-valid, in-vocabulary choice is produced within ``retries``.
        """
        options = list(options)
        schema = {
            "type": "object",
            "properties": {
                "choice": {"type": "string", "enum": options},
                "reason": {"type": "string"},
            },
            "required": ["choice", "reason"],
            "additionalProperties": False,
        }
        last = None
        for _ in range(retries + 1):
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "survey_answer", "strict": True, "schema": schema},
                },
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=max_tokens,
                extra_body={"reasoning": {"enabled": self.config.reasoning}},
            )
            last = resp.choices[0].message.content
            try:
                data = json.loads(last)
            except (json.JSONDecodeError, TypeError):
                continue
            if data.get("choice") in options:
                return {
                    "choice": data["choice"],
                    "reason": data.get("reason", ""),
                    "model": self.config.model,  # pinned & logged (R6)
                    "usage": resp.usage.model_dump() if resp.usage else None,
                }
        raise LLMError(f"No valid choice after {retries + 1} tries; last={last!r}")
