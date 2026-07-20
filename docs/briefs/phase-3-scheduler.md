# Brief — Phase 3: Scheduling & throughput

**Phase:** 3 · **Rules activated:** R5 (per-agent-per-tick cache scoping), R6 (model pinned & logged per run) · **Layer:** Architecture / Engine.

## Context

The Phase 2 tick loop (`simulation.py`) is single-threaded: it calls the persona
endpoint once per agent per tick, serially, with no concurrency, no retry, and no
latency/cost accounting. Phase 3's job is to make throughput at 20–30 agents
sustainable and *measured* before Phase 6 scales to 100.

**Premise challenged (ADR 0009).** The phase plan's spike names *"batched local
inference on home hardware."* But the baseline persona model is `qwen/qwen3-32b`
(ADR 0005), whose AWQ 4-bit weights (~18–20 GB) do not fit the 16 GB RTX 4070 Ti
SUPER, and ADR 0002 deliberately keeps the persona LLM on managed **OpenRouter**
until P5 (self-hosted vLLM on rented GPU). On a managed endpoint the throughput
lever is **request concurrency**, not local GPU continuous-batching. So Phase 3
builds a **concurrent scheduler against the managed backend** and benchmarks *it*;
the vLLM continuous-batching path is designed as a seam and its benchmarking is
deferred to P5 where self-hosting is real. R5 is honored **structurally** — a
prohibition ("never share reasoning across agents"), not a mandate to run a local
KV cache now.

**Concurrency ∩ R28.** Only the *decide* phase of a `simultaneous` tick is
parallelizable: every agent decides from the same pre-tick snapshot, reading only
its own private memory (R2), so the calls are independent. The resolve/apply phase
(the sole writer of world + cross-agent memory + run log) stays serial, preserving
the run log's single-writer invariant (ADR 0006). `sequential` ticks are serial by
definition (each agent must see prior applies) and are timed but not parallelized.

## Definition of done

- 20–30 agents run multiple ticks **without failure**, with per-call **latency**
  and **token/cost** logged, on `qwen/qwen3-32b` via OpenRouter
  (`notebooks/experiments/phase3_throughput.ipynb`).
- The decide phase of a simultaneous tick runs concurrently with **bounded**
  in-flight requests; concurrency is a run parameter (R17) and never merges/shares
  reasoning across agents (R5).
- Transient endpoint failures (rate-limit/timeout) are retried with backoff; the
  run completes rather than dying on the first 429.
- Model + provider are pinned and logged in the run config (R6); token usage +
  latency are on each action event; a per-tick throughput summary is logged
  (`tick_metrics`); a run-level throughput aggregate is returned on `Run`.
- Deterministic test suite green, including the P2 effect/log-consistency tests
  (the serial-writer invariant must still hold under the concurrent decide phase).

## Prerequisites

Phase 2 complete (branch `phase-2-game-master`). `.venv` (Python 3.11).

## Ordered tasks

1. `src/polis/scheduler.py` — `SchedulerConfig`, `Timing`, `Scheduler.map()` (bounded
   ThreadPool, retry/backoff, per-unit timing), `estimate_cost_usd()`. TDD.
2. `src/polis/actions.py` — `ActionDecision` gains optional `usage`/`model`.
3. `src/polis/agent.py` — `act()` populates `usage`/`model` from the decode.
4. `src/polis/runlog.py` — add `EVENT_TICK_METRICS`.
5. `src/polis/simulation.py` — concurrent simultaneous decide phase via the
   scheduler; enrich action events (latency/tokens/attempts); log per-tick + return
   run-level throughput; add provider/base_url + scheduler block to the run config.
6. Extend `tests/test_simulation.py`; add `tests/test_scheduler.py`.
7. `notebooks/experiments/phase3_throughput.ipynb` — the DoD benchmark + a
   concurrency sweep; results to `data/phase3_throughput.csv`.

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest -q` green (deterministic).
- Scheduler test proves: order preserved, concurrency bounded by `max_concurrency`,
  wall-time under serial sum, retries counted, latency captured.
- `test_simulation.py` proves the P2 memory/world/log consistency is unchanged and
  the new throughput events/aggregate are present.
- Benchmark notebook runs 20–30 agents × ≥3 ticks live, no failures, latency/cost
  logged.

## Hand-off pointer

Update `status.md`, `PHASE_PLAN.md` status, write `docs/checkpoints/phase-3.md`,
record ADR 0009. The vLLM batch-executor seam and the log's single-writer
assumption are the notes P5/P6 inherit.
