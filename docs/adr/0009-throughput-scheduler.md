# ADR 0009: Phase 3 throughput — concurrent managed scheduler, not local GPU batching

Status: accepted
Phase: 3

## Spike question

What agent count / tick rate is sustainable on home hardware with batched local
inference (R5/R6)?

## Context

The Phase 2 tick loop calls the persona endpoint once per agent per tick, serially,
with no concurrency, retry, or latency/cost accounting. Phase 3's job is to make
20–30 agents over multiple ticks sustainable and *measured* before P6 scales to 100.

The phase plan framed this as *"benchmark vLLM (or Ollama) batching on home
hardware."* Three repo facts collide with that literal framing:

1. **The persona model can't run locally on this box.** Baseline is `qwen/qwen3-32b`
   (ADR 0005). AWQ 4-bit weights (~18–20 GB) exceed the 16 GB RTX 4070 Ti SUPER
   before any KV cache — which is *why* P5 targets rented GPU (RunPod/Koyeb), not
   the local card, and why CLAUDE.md reserves that card for lightweight models
   (embeddings).
2. **ADR 0002 keeps the persona LLM on managed OpenRouter until P5**, precisely
   because P0–4 validate mechanics, not reported data. Self-hosting now pulls
   forward work that ADR deliberately deferred.
3. **On a managed endpoint the throughput lever is request concurrency, not local
   GPU continuous-batching** — we don't control OpenRouter's batching or KV cache.

R5 ("warm/cold cache scoped strictly per-agent-per-tick; never merge/share cached
reasoning across agents") is a **prohibition**, satisfiable structurally — it does
not mandate operating a local KV cache now.

## Options considered

1. **Concurrent scheduler against the managed backend (chosen).** A per-agent-per-
   tick concurrent decide phase (bounded parallelism, retry/backoff, latency+token+
   cost logging) benchmarked on OpenRouter `qwen/qwen3-32b`. The vLLM continuous-
   batching path is a designed but unimplemented executor seam, benchmarked at P5.
   Honest numbers on the backend we actually use; respects ADR 0002. Cost: the
   home-hardware GPU-batching question is answered later, not now.
2. **Pull P5 self-host forward — benchmark local GPU batching literally.** 32B-AWQ
   won't fit 16 GB, so this benchmarks a smaller model (8B/14B) whose throughput
   won't transfer to P5's 32B-on-rented-GPU runs; contradicts ADR 0002 and the
   "local card = embeddings only" rule. Rejected: a non-transferable number.
3. **Hybrid — managed scheduler + a local small-model smoke.** Adds a throwaway
   local benchmark beside the real one. Rejected: extra work for a number labelled
   non-transferable anyway.

## Decision

**Option 1.** `src/polis/scheduler.py` adds a `Scheduler` that runs per-agent-per-
tick decide **units** concurrently — a bounded `ThreadPoolExecutor`
(`max_concurrency`), each unit retried with capped exponential backoff — returning
`(key, result, Timing)` in input order. It is LLM-agnostic (times/retries/gathers
any callable), so it is reused by the sequential path (`run_with_retry`) and is the
slot the P5 vLLM batch executor drops into. `SchedulerConfig(executor="batch")` is
rejected at construction with a "P5 seam" error — declared, not half-wired.

Only the **decide phase of a `simultaneous` tick** is parallelized: every agent
decides from the same pre-tick snapshot reading only its own private memory (R2),
so the calls are independent. The resolve/apply phase (sole writer of world +
cross-agent memory + run log) stays serial, preserving the run log's single-writer
invariant (ADR 0006). `sequential` ticks are serial by definition (R28) but timed
and logged identically.

Latency + token usage are logged on each `action` event; a per-tick `tick_metrics`
event carries the throughput summary (R15-style trajectory); `Run.throughput` is
the run-level aggregate (`decides_per_s`, latency mean/p95/max, tokens, retries,
failures, optional USD estimate) — kept **distinct** from `Run.metrics` so the
infra signal and the P5 convergence signal never conflate. Model + provider
(base_url) and the scheduler knobs are pinned into the run config (R6/R17).

An unrecoverable decide (retries exhausted) **raises** and aborts the run rather
than silently degrading to ABSTAIN — a throughput failure is a finding, and a
silent abstain would distort dynamics.

## Why

The value Phase 3 must deliver — sustainable throughput and failure-handling at
20–30 agents, with logged latency/cost, before scaling to 100 — is real and lands
now. The specific mechanism the plan named (local GPU batching) is infeasible for
this model on this hardware and premature per ADR 0002. Concurrency is the correct
throughput lever for a managed endpoint, and R5 is honoured by *construction* (the
scheduler caps in-flight calls and shares no prompt/state/cache across units). This
amends the Phase 3 framing in `PHASE_PLAN.md`; it does not change the DoD (20–30
agents, multiple ticks, logged latency/cost), which is backend-agnostic as written.

If, mid-debug at P5/P6, throughput numbers look wrong, this decision is a plausible
root cause to check: the numbers here are OpenRouter-network-bound concurrency, not
GPU-batching throughput — they set an ordering/latency expectation, not a
GPU-utilisation one.

## Consequences

- **The home-hardware GPU-batching benchmark is deferred to P5**, when self-hosted
  vLLM on rented GPU is real. The `executor="batch"` seam + the query handbook's
  vLLM guided-decoding notes are where it lands. P3's numbers are the *managed*
  baseline it will be compared against.
- **The run log's single-writer assumption (ADR 0006) still holds**: concurrency is
  confined to the read-only decide phase; all log writes remain on the main thread.
  Revisit if a future phase parallelizes the *apply* phase or the log writer.
- **`sequential` scheme gains no throughput** from this work — it is inherently
  serial (R28). Concurrency and the update scheme interact: only `simultaneous`
  fans out. Recorded so a P4/P5 run doesn't expect otherwise.
- **Cost is logged as token counts (real) plus an optional USD estimate** from a
  configurable `price_per_mtok` rate (default `None` → no guessed price). Provider-
  authoritative cost can be wired later if OpenRouter's per-call cost field is
  needed; tokens are the durable figure.
- Adds `EVENT_TICK_METRICS` to the closed run-log vocabulary; `ActionDecision`
  gains optional `usage`/`model`; `Simulation` gains a `scheduler`/
  `scheduler_config` seam. No change to `choose()`/`decide()` contracts.

## Rules touched

R5 (per-agent-per-tick scheduling scope — cap, never share), R6 (model + provider
pinned & logged per run). Constrained: R28 (scheme ∩ concurrency), R15 (per-tick
throughput trajectory), R17 (scheduler knobs versioned). Amends the Phase 3 framing
in `PHASE_PLAN.md`; consistent with ADR 0002 (provider phasing) and ADR 0006
(single-writer run log).
