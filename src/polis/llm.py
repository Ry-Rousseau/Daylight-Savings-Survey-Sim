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
import time
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
# Baseline model: Qwen3-32B (ADR 0005, amending ADR 0002's 8B pin). The P1
# model-capability sweep found 8B needs hand-annotated survey options to answer
# on the merits; 14B/32B do not. Model id is a per-call field (R1) and logged (R6).
DEFAULT_MODEL = "qwen/qwen3-32b"


class LLMError(RuntimeError):
    """Raised when the endpoint fails to return a usable, in-vocabulary answer."""


def retry_on_llm_error(fn, *, attempts: int = 3, backoff: float = 0.5):
    """Call ``fn`` (a structured LLM call), retrying on :class:`LLMError` with linear
    backoff. Returns ``fn()``'s result, or ``None`` if it never succeeds within
    ``attempts`` — so one flaky agent can be *skipped* rather than aborting a whole
    batch (the survey fan-out, the drift probe). Only ``LLMError`` is caught; real
    bugs propagate. The endpoint's own soft json_schema enforcement means any model
    can occasionally miss, so this resilience is model-agnostic, not Qwen-specific."""
    for i in range(attempts):
        try:
            return fn()
        except LLMError:
            if i + 1 < attempts:
                time.sleep(backoff * (i + 1))
    return None


@dataclass(frozen=True)
class LLMConfig:
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.8  # per-agent generation param (R1)
    top_p: float = 1.0
    max_tokens: int = 512
    # Structured (json_schema) calls get their own budget: a verbose or reasoning model
    # can truncate a JSON object mid-object under a tight cap, which then fails to parse.
    # Separate from ``max_tokens`` so the free-form ``complete`` path is unaffected.
    structured_max_tokens: int = 512
    # Provider-specific "thinking"/reasoning toggle (OpenRouter forwards it to Qwen3).
    # ``True``/``False`` sends the toggle; ``None`` omits the param entirely, so swapping
    # to a model/backend that doesn't understand it needs no code change here.
    reasoning: bool | None = False


class LLMClient:
    """Thin OpenAI-compatible client pointed at OpenRouter."""

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None):
        self.config = config or LLMConfig()
        load_dotenv(find_dotenv(usecwd=True))
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise LLMError("OPENROUTER_API_KEY is not set (add it to .env).")
        self._client = OpenAI(base_url=self.config.base_url, api_key=key)

    def _extra_body(self) -> dict | None:
        """Provider-specific extras. The reasoning toggle is sent only when configured
        (``None`` => omitted), so a non-Qwen model isn't handed a param it doesn't
        understand. ``None`` return means 'no extra body' to the OpenAI client."""
        if self.config.reasoning is None:
            return None
        return {"reasoning": {"enabled": self.config.reasoning}}

    def complete(self, messages, *, temperature=None, max_tokens=None) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=self.config.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            extra_body=self._extra_body(),
        )
        return resp.choices[0].message.content

    def _structured_call(self, *, system, user, schema, schema_name, validate,
                         temperature=None, max_tokens=None, retries=1):
        """Shared json_schema call + parse + validate + retry (soft-enforced on
        OpenRouter; hard grammar arrives with vLLM at P5, ADR 0002). Returns
        ``(data, resp)`` for the first response that parses and passes
        ``validate``; raises ``LLMError`` if none does within ``retries``.

        ``max_tokens=None`` falls back to ``config.structured_max_tokens`` so a
        verbose/reasoning model gets enough room not to truncate its JSON."""
        last = None
        tokens = self.config.structured_max_tokens if max_tokens is None else max_tokens
        for _ in range(retries + 1):
            resp = self._client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "strict": True, "schema": schema},
                },
                temperature=self.config.temperature if temperature is None else temperature,
                max_tokens=tokens,
                extra_body=self._extra_body(),
            )
            last = resp.choices[0].message.content
            try:
                data = json.loads(last)
            except (json.JSONDecodeError, TypeError):
                continue
            if validate(data):
                return data, resp
        raise LLMError(f"No valid {schema_name} after {retries + 1} tries; last={last!r}")

    def choose(self, *, system, user, options, temperature=None, max_tokens=None, retries=1) -> dict:
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
        data, resp = self._structured_call(
            system=system, user=user, schema=schema, schema_name="survey_answer",
            validate=lambda d: d.get("choice") in options,
            temperature=temperature, max_tokens=max_tokens, retries=retries,
        )
        return {
            "choice": data["choice"],
            "reason": data.get("reason", ""),
            "model": self.config.model,  # pinned & logged (R6)
            "usage": resp.usage.model_dump() if resp.usage else None,
        }

    def decide(self, *, system, user, schema, valid_types, temperature=None,
               max_tokens=None, retries=1) -> dict:
        """Structured action decode for the tick loop (R20/R23). Validates only
        that ``action_type`` is in the closed vocabulary; payload validity
        (a well-formed SPEAK) is the Game Master's call (R24), so a payload-light
        action is returned as-is and degrades to a no-op downstream rather than
        forcing a retry. Returns the parsed action dict plus ``model``/``usage``.
        """
        data, resp = self._structured_call(
            system=system, user=user, schema=schema, schema_name="agent_action",
            validate=lambda d: d.get("action_type") in valid_types,
            temperature=temperature, max_tokens=max_tokens, retries=retries,
        )
        return {
            **data,
            "model": self.config.model,  # pinned & logged (R6)
            "usage": resp.usage.model_dump() if resp.usage else None,
        }
