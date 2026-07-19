# Brief — Phase 0: Walking skeleton

## Context
`polis` is a census-seeded LLM-persona "silicon sample" of NYC, surveyed on daylight saving time, with convergence treated as a layered cross-cutting risk (`design_layers.md`). Per the phase plan (`implementation_rough_plan.md`), we build a walking skeleton and thicken one axis at a time. Phase 0 is the thinnest runnable slice: prove the single hardest architectural unknown — that the local model can be driven to return **schema-valid structured output** and that a **minimal LangGraph fan-out/gather** works — before any memory, interaction, or population scale exists.

## Spike questions (→ resolve as ADRs in `docs/adr/`)
- Does local vLLM + constrained decoding reliably return **schema-valid JSON** for a survey answer?
- Does a minimal **LangGraph fan-out/gather** work for 3 agents (parallel calls, gathered responses)?

## Definition of done
**3 hardcoded agents answer one hardcoded survey question, returning 3 parseable (schema-valid) responses.** No memory, no interaction, no census seeding — structured output only.

## Prerequisites
- Scaffold + conventions + spec docs read. ✓
- Python env reconciled to 3.12 (open thread) and deps installed: LangGraph, an OpenAI-compatible client, pydantic, pytest.
- vLLM server up: `Qwen/Qwen3-8B-AWQ` at `http://localhost:8000/v1` (see ADR 0001 for the launch command).

## Ordered tasks
1. Reconcile the Python env (3.11 vs 3.12); create/confirm `.venv`; add `pyproject.toml` making `polis` importable + pytest configured.
2. Stand up the vLLM server; smoke-test a raw completion against `localhost:8000`.
3. Define the survey-answer schema (single-select) as a pydantic model; wire grammar-constrained decoding so output is schema-valid by construction.
4. In `src/polis/`, implement the minimal agent call (prompt → constrained JSON → parsed answer) with a test.
5. Add a LangGraph graph that fans out to 3 hardcoded agents and gathers their answers.
6. Run end-to-end: 3 agents, 1 question, 3 parseable responses; assert in a test.

## Acceptance checks
- `pytest` green: agent call returns a schema-valid, parsed answer; the 3-agent fan-out gathers 3 responses.
- A scripted run prints the 3 agents' answers to one DST question.
- Model identity/version is logged with the run (R6); generation params are per-agent fields, not globals (R1).

## Rules in play
- **Build this phase:** R1 (per-agent gen params), R20 (structured output — pending definition in `design_layers.md`).
- **Standing constraints:** R2/R3/R5 (state separation, no shared cached reasoning) — don't violate as later features land.

## Hand-off pointer
Next: **Phase 1 — Memory** (vector store + recency/importance/relevance; two agents with different seeded memories give measurably different answers). See `implementation_rough_plan.md` Phase 1.
