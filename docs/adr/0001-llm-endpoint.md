# ADR 0001: Remote vLLM (Qwen3-8B-AWQ) as the persona LLM backend

Status: accepted (amended by ADR 0002 — phased: OpenRouter for P0–2, self-hosted vLLM for P5+)
Phase: 0

## Spike question

Which LLM endpoint backs the personas, given R6 (pin & log model per run) and the need to fan out across many agents?

## Context

The workload is ~100 personas × multi-tick interaction × repeated surveys → high call volume with heavy KV-cache pressure. The model is served by **vLLM on a remote GPU host** (RunPod / Koyeb / similar) exposing an **OpenAI-compatible** API; we call it from the local machine via the OpenAI SDK. The developer's local RTX 4070Ti SUPER (16 GB) is **not** running the persona model — it stays free for lightweight local models (e.g. the Phase 1 embedding model). `requirements_features.md` #2 calls for schema-constrained decoding to compile agent intent into a fixed action vocabulary; convergence rules require model identity/version pinned and logged per run (R6) and generation params exposed per-agent, not global (R1).

## Options considered

1. **Remote vLLM serving `Qwen/Qwen3-8B-AWQ` (AWQ), OpenAI-compatible** — a proper GPU without local-hardware limits; PagedAttention pages the KV cache on demand so many concurrent agent contexts fit without fragmentation waste; native grammar-constrained decoding (`guided_json` / `guided_choice`); cheap per call. Cost: an ephemeral provider-supplied host (cold starts, teardown), an 8B-class quality ceiling, single-endpoint throughput cap.
2. **Hosted Claude / OpenAI** — higher quality, zero infra; but per-call cost scales badly with agents × ticks, and constrained output is via tool/JSON mode rather than grammar.
3. **Local model on the 4070Ti** — free, but 16 GB caps concurrency for a many-agent fan-out; better reserved for a local embedding model.

## Decision

Option 1 — remote vLLM / `Qwen/Qwen3-8B-AWQ`, OpenAI-compatible at `http://<host>:8000/v1`, accessed via the OpenAI SDK (`base_url` = the provider host, `api_key="not-needed"` unless the server sets `--api-key`). Setup/query details and the parameter reference live in `docs/query_handbook.md`.

## Why

Agent-count fan-out and cost dominate this workload. PagedAttention allocates KV-cache pages on demand instead of pre-allocating a contiguous block per request, so many concurrent agent contexts coexist on one GPU — exactly this workload's shape. A remote host gives that GPU without local constraints; cheap per call means runs iterate without spend pressure; and grammar-constrained decoding is native for the structured-output layer.

## Consequences

Locks in an OpenAI-compatible client and an 8B-class quality ceiling. The host URL is **ephemeral and provider-supplied** → treat `base_url` as config (env `POLIS_LLM_BASE_URL`), never hardcode; expect cold starts / teardown. Model is a per-agent field (R1), so agents can be swapped later. **Watch:** vLLM server-side prefix caching may share identical prompt prefixes across requests — fine for a genuinely shared, logged environment signal, but must not become shared *private* reasoning across agents (R3/R5). Revisit if: agent count/context exceeds single-endpoint throughput (Phase 3 benchmark); persona fidelity proves insufficient at 8B; or multi-model ablations are needed for baseline-convergence comparisons (R6/R16).

## Rules touched

R1 (per-agent generation params), R3/R5 (no shared cached reasoning across agents), R6 (pinned/logged model identity).
