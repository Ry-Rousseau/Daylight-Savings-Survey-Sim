# ADR 0003: Phase 0 spike resolutions — constrained output + LangGraph fan-out

Status: accepted
Phase: 0

## Spike question

(1) Does the model + constrained decoding reliably return schema-valid JSON? (2) Does a minimal LangGraph fan-out/gather work for 3 agents?

## Context

Phase 0 DoD: 3 hardcoded agents answer one survey question, 3 parseable responses. Backend is OpenRouter `qwen/qwen3-8b` (ADR 0002).

## Findings / Decision

1. **Constrained output:** `response_format` json_schema returns clean, in-vocabulary answers **only with Qwen3 reasoning disabled** (`extra_body={"reasoning": {"enabled": False}}`). With reasoning on, output degenerated into unclosed JSON (whitespace runaway to the token cap). OpenRouter's json_schema is soft-enforced, so `LLMClient.choose()` validates `choice ∈ options` and retries once, raising `LLMError` otherwise.
2. **Orchestration:** a LangGraph `Send` map from `START` → per-agent `ask` node, gathered via an `operator.add` reducer on `answers`, works for N agents (`polis.graph.run_survey`). This is the query-layer tool only (R22); the sim tick loop stays a separate custom loop.

## Consequences

- The `choose()` contract — returns `{choice, reason, model, usage}` — is stable for later phases; agents gain a memory-retrieval step *before* it in P1.
- Hard grammar enforcement (R20) is deferred to self-hosted vLLM at P5+ (ADR 0002); until then, validate-and-retry is the guarantee.

## Rules touched

R20 (constrained output), R22 (two-tool orchestration), R1/R6 (per-agent params; pinned model returned for logging).
