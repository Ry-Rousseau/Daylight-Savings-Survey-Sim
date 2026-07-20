# Checkpoint — Phase 3: Scheduling & throughput

**Date:** 2026-07-20 · **Status:** ✅ complete (DoD met) · **Branch:** built on the P2 tree (uncommitted)

## What it proves

The Phase 2 tick loop called the endpoint once per agent per tick, serially, with
no concurrency/retry/accounting. Phase 3 adds a per-agent-per-tick **concurrent
scheduler** (R5) and full latency/token/cost logging (R6), so 20–30 agents over
multiple ticks are sustainable and *measured*. The premise was challenged first
(ADR 0009): local GPU batching of the 32B persona model is infeasible on the 16 GB
card and premature per ADR 0002, so P3 benchmarks **request concurrency against
managed OpenRouter**; the vLLM continuous-batching path is a designed seam
benchmarked at P5.

**Live DoD run** (`qwen/qwen3-32b`, 25 agents × 3 ticks = 75 decides per config,
simultaneous scheme, `notebooks/experiments/phase3_throughput.ipynb`): **zero
failures, zero retries** across a concurrency sweep, with per-call latency + tokens
logged. Throughput scales cleanly with the concurrency cap while mean latency stays
roughly flat (the OpenRouter endpoint is nowhere near saturated at these caps):

| concurrency | wall (s) | decides/s | latency mean / p95 / max (s) |
|---:|---:|---:|---|
| 1  | 316.7 | 0.24 | 3.98 / 9.94 / 11.6 |
| 4  | 112.7 | 0.67 | 4.32 / 12.5 / 15.5 |
| 8  | 82.4  | 0.91 | 5.72 / 10.4 / 22.5 |
| 16 | 55.3  | 1.36 | 4.36 / 8.36 / 12.0 |

**5.7× throughput** from c=1 → c=16 (0.24 → 1.36 decides/s), the answer to the
spike question: at 25 agents on OpenRouter, sustainable tick rate is
concurrency-bound, ~1.4 decides/s at c=16 (a 75-decide, 3-tick run in ~55 s).
~43k tokens/config (≈35k prompt / 7.5k completion); `est_cost_usd` null by design
(`PRICE_PER_MTOK` unset — tokens are the logged truth). Results in
`data/phase3_throughput.csv`.

Deterministic proof (no network): **53 tests pass** (33 from P0–P2 + 20 new — 12
scheduler, 8 simulation). The P2 effect/log-consistency tests still pass *unchanged*
under the concurrent decide phase, confirming the run log's single-writer invariant
(ADR 0006) holds.

## What's live

- `src/polis/scheduler.py` — `Scheduler.map()` runs per-agent-per-tick decide units
  on a bounded `ThreadPoolExecutor` (`max_concurrency`, R5), each retried with
  capped exponential backoff; returns `(key, result, Timing)` in input order.
  LLM-agnostic (reused by the sequential path via `run_with_retry`); `executor=
  "batch"` is a P5 seam rejected at construction. `estimate_cost_usd()` (tokens ×
  optional rate). ADR 0009.
- `src/polis/simulation.py` — the `simultaneous` decide phase fans out through the
  scheduler (independent per-agent reads, R2); resolve/apply stays serial
  (single-writer). Action events now carry `model`/`latency_s`/`attempts`/token
  counts; a per-tick `EVENT_TICK_METRICS` summary is logged (R15 trajectory);
  `Run.throughput` is the run-level aggregate (`decides_per_s`, latency mean/p95/
  max, tokens, retries, failures, `est_cost_usd`) — distinct from the P5
  `Run.metrics` divergence stub. Run config pins provider/base_url + scheduler knobs
  (R6/R17). An exhausted decide raises (fail loud), never silent-abstains.
- `src/polis/actions.py` — `ActionDecision` gains optional `usage`/`model`.
- `src/polis/agent.py` — `act()` surfaces the decode's `usage`/`model`.
- `src/polis/runlog.py` — `EVENT_TICK_METRICS` added to the closed vocabulary.
- Tests: `tests/test_scheduler.py` (order, bounded concurrency, retry/backoff,
  timing, cost, config guards); `tests/test_simulation.py` extended (concurrent
  effect/log consistency at N=10, latency/token action fields, per-tick metrics,
  run aggregate, provider/scheduler config pinning, retry-then-succeed, unrecoverable
  abort, sequential throughput).
- `notebooks/experiments/phase3_throughput.ipynb` — the DoD benchmark: one run +
  a concurrency sweep `[1,4,8,16]` at 25 agents × 3 ticks, latency/cost logged,
  results → `data/phase3_throughput.csv`, plots + per-tick trajectory. Valid nbformat;
  glue dry-run verified. **Awaiting a live execution.**

## DoD status

| DoD clause | State |
|---|---|
| 20–30 agents run multiple ticks **without failure** | ✅ 25 agents × 3 ticks × 4 configs, 0 failures live |
| **latency logged** per call/tick/run | ✅ built + live-verified |
| **cost logged** (tokens always; USD when rate set) | ✅ ~43k tokens/config logged; USD null (rate unset) |
| concurrency bounded & never shares reasoning (R5) | ✅ built + tested; 5.7× scaling c=1→16 |
| model + provider pinned/logged (R6/R17) | ✅ built + tested |
| deterministic suite green (serial-writer invariant intact) | ✅ 53 passed |

## What P4/P5/P6 need

- **P5 (self-host):** the `executor="batch"` seam + the query handbook's vLLM
  guided-decoding notes are where local-GPU continuous batching gets benchmarked —
  against P3's managed numbers as the baseline. R5's "never share KV/reasoning
  across agents" becomes load-bearing there in a way it can't be on OpenRouter.
- **P6 (scale to 100):** re-run the sweep at N=100; watch whether `decides_per_s`
  and rate-limit/retry behaviour hold, and whether the serial resolve/apply +
  single-writer log become the bottleneck (they were fine at 30).
- **Standing:** if a future phase parallelizes the *apply* phase or the log writer,
  the single-writer assumption (ADR 0006) is void — revisit then.

## Debt / notes

- **Sequential scheme gains no throughput** — inherently serial (R28); only
  `simultaneous` fans out. Recorded so P4/P5 don't expect otherwise.
- **Cost is an estimate** unless `price_per_mtok` is set (default `None`); tokens are
  the durable truth. Provider-authoritative per-call cost can be wired later.
- **Concurrent embedding encode:** under the threaded decide phase, agents sharing
  one `EmbeddingModel` call `.encode()` concurrently; correct, but GPU-serialized —
  a minor, non-R5 caveat (embeddings are the *local* model, not the persona cache).
