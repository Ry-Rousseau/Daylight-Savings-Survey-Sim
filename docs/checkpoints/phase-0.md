# Checkpoint — Phase 0: Walking skeleton

**Date:** 2026-07-19 · **Status:** ✅ complete

## What it proved
- End-to-end path works: 3 hardcoded personas answer one DST survey question and return 3 schema-valid, in-character responses (`python -m polis`).
- Structured single-select via `response_format` json_schema is reliable **once Qwen3 reasoning is disabled** — otherwise the model derails into unclosed JSON (see ADR 0003).
- A minimal LangGraph `Send` fan-out/gather over 3 agents works (`tests/test_graph.py`).

## What's live
- `src/polis/`: `survey` (schemas), `llm` (OpenRouter client + `choose()`), `persona` (3 seed personas), `agent`, `graph` (fan-out/gather), `__main__` (demo).
- Tests: 5 passing — `test_survey`, `test_graph` (no network), `test_llm_live` (live, skipped without key), `test_smoke`.
- Endpoint: OpenRouter `qwen/qwen3-8b`, key in `.env` (ADR 0001 + 0002).

## Observed (sanity only — not a result)
2 personas → permanent DST, 1 → permanent standard time. Divergence present, personas voice distinct interests. No metrics, no validation yet — this is mechanics, not data.

## What P1 (Memory) needs
- A per-agent memory store the agent reads before answering, so seeded memories measurably change answers (the P1 DoD).
- Embedding model choice (candidate: local on the 4070Ti) + recency/importance/relevance scoring.
- `choose()` stays the output contract; the agent gains a retrieval step before it.

## Debt / notes
- OpenRouter json_schema is *soft*-enforced → validate + retry (`LLMError` on failure). Hard grammar (R20) returns with self-hosted vLLM at P5+ (ADR 0002).
- No `abstain` option yet (R25) — add when interaction/actions arrive (P2).
- Console mojibake for non-ASCII on Windows cp1252 — cosmetic; data is correct UTF-8.
