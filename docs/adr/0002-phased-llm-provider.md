# ADR 0002: Phased LLM provider — OpenRouter now, self-hosted vLLM at validation

Status: accepted (model pin amended by ADR 0005 — baseline is now `qwen/qwen3-32b`; the provider phasing below is unchanged)
Phase: 0

## Spike question

Given ADR 0001 chose "remote vLLM", what do we actually run for the early disposable phases vs the load-bearing ones?

## Context

Phases 0–2 (walking skeleton, memory, Game Master) validate mechanics, not data we report on — R6/R17 rigor doesn't bite yet. Phases 5+ produce the runs that feed divergence metrics and any calibration check, where model/version pinning (R6) and config-versioned runs (R17) are load-bearing. Standing up a vLLM box is trivial effort relative to those later phases.

## Options considered

1. **Self-hosted vLLM from day 0** — full control/rigor immediately, but setup + idle cost during throwaway phases.
2. **Managed provider throughout** — zero setup, but no pinned commit/quantization for the reportable runs (weakens R6/R17).
3. **Phased: managed (OpenRouter) for P0–2, self-hosted vLLM for P5+.**

## Decision

Option 3. **P0–2:** OpenRouter, OpenAI-compatible (`https://openrouter.ai/api/v1`, model `qwen/qwen3-8b`, key `OPENROUTER_API_KEY` in `.env`). **P5+:** self-hosted vLLM (`Qwen/Qwen3-8B-AWQ`) on a RunPod Pod/Serverless (or local hardware if the 8B-Q4 math fits), with a pinned commit + quantization.

## Why

Cheap, zero-setup, fast iteration wins outright while phases are disposable; full control matters exactly when the output becomes data you report on. Both endpoints are OpenAI-compatible, so the client code (`src/polis/llm.py`) is unchanged across the switch — only `base_url`, the model id, and the structured-output mechanism differ.

## Consequences

- P0–2 structured output is *soft*-enforced (json_schema + validate/retry); the hard grammar guarantee (R20) arrives with vLLM guided decoding at P5+.
- Qwen3 "thinking" is disabled for survey answers — a JSON schema + reasoning derails output, and it's cheaper (see ADR 0003).
- Amends ADR 0001 (which assumed remote vLLM as the single endpoint). Revisit the switch point if a mechanics phase starts needing pinned-model rigor earlier.

## Rules touched

R1 (per-agent params), R6 (pinned/logged model), R17 (config-versioned runs), R20 (constrained output).
