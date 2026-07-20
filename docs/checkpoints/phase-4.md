# Checkpoint — Phase 4: Topology

**Date:** 2026-07-20 · **Status:** ✅ complete (DoD met) · **Branch:** `phase-4-topology` (uncommitted)

## What it proves

Through Phase 3 the interaction graph was a single hardcoded seam — `fully_connected`,
the R10 anti-pattern (everyone hears everyone). Phase 4 makes topology a **pluggable,
seeded, swappable** parameter (R4/R13) with sparse/clustered alternatives (R10), and
shows that swapping it — same personas, different graph — produces a **starkly
different homogeneity metric**, the Phase-4 spike. The metric itself is the P4-scoped
instrument in `src/polis/metrics.py` (dominant-share / distinct / normalised entropy
over the population's SPEAK stances), defined and unit-tested *before* the run per R14
(ADR 0012). Premise challenged first: the DoD needs a metric but R14–R17 is a P5
deliverable, so P4 builds the minimal tested kernel P5 extends, not the full dashboard.

**Live DoD run** (`qwen/qwen3-32b`, 20 agents [10 evening / 10 morning seeds] × 4
ticks, `notebooks/experiments/phase4_topology.ipynb`, **0 failures** across configs):

| topology | dom-share trajectory (t0→t3) | final entropy | survey dom-share |
|---|---|---:|---:|
| `fully_connected` | 0.50 → **1.00** → 1.00 → 1.00 | 0.00 | 1.00 |
| `small_world` (k=4, p=0.2) | 0.50 → 0.50 → 0.50 → **0.55** | 0.50 | 0.60 |
| `stochastic_block` (2 blocks, p_in 0.7 / p_out 0.03) | 0.50 → 0.50 → 0.50 → **0.55** | 0.50 | 0.50 |

Full connectivity **collapses to unanimity by tick 1** (dominant share 1.00, entropy
0.00 — everyone on permanent DST); the sparse and clustered graphs **hold the initial
split** for the whole run (0.55 / entropy ≈ 0.50). Full exposure biases toward global
consensus exactly as R10 predicts. Endpoint surveys (the Layer-5 op, R18) agree with
the logged-stance trajectory. Results in `data/phase4_topology.csv`.

**Committed minority (R11) — the opposite of the naïve intuition.** The identical
4-agent faction committed to *permanent standard time* persuades **0%** of the
non-committed agents under `fully_connected` but **31%** under `stochastic_block`.
Full connectivity does not amplify the faction — the same global-consensus cascade
that produced unanimity in §2 **drowns** four fixed voices; the clustered graph
**shelters** the faction inside its block, where it keeps local influence. R10 + R11
in one comparison: sparse/clustered topology preserves a committed minority against a
global majority.

Deterministic proof (no network): **89 tests pass** (53 from P0–P3 + 36 new — 14
topology, 11 metrics, 11 simulation additions). The P2/P3 effect/log-consistency tests
still pass unchanged under the new topologies.

## What's live

- `src/polis/topology.py` — `Topology` base (callable `(agent_id, roster) -> listeners`
  + cached-frozen symmetric adjacency + `to_config()` for R17) and four seeded,
  deterministic graphs: `FullyConnected`, `RingLattice(k)`, `SmallWorld(k, p)`
  (Watts–Strogatz), `StochasticBlock(n_blocks, p_in, p_out)` (with
  `block_assignments()`). Drop-in for the existing callable seam. ADR 0011.
- `src/polis/metrics.py` — `homogeneity(distribution, support=)`,
  `stance_distribution(run, tick=)`, `homogeneity_trajectory(run)`. Reads the `action`
  event stream offline; the shared kernel P5's full R14 layer extends. ADR 0012.
- `src/polis/simulation.py` — `DynamicsConfig.exchange_volume` (R12: cap a SPEAK's
  reach, sampled deterministically from `(seed, tick, agent_id)`); `_resolve` applies
  it; run config now versions the structured topology descriptor, `exchange_volume`,
  and the committed roster (R17).
- `src/polis/agent.py` — `Agent.committed_stance` + `committed()` helper: an immovable
  faction agent that SPEAKs a fixed stance with **no model call** and empty provenance
  (R11).
- `src/polis/runlog.py` — `EVENT_TIE_CHANGE` reserved as the R26 seam (not emitted).
- Tests: `tests/test_topology.py` (symmetry, seeded reproducibility, k-regularity,
  sparsity ordering, SBM intra>inter density, frozen-after-build, param validation);
  `tests/test_metrics.py` (homogeneity math, log read, trajectory); `tests/test_simulation.py`
  P4 additions (committed no-model-call, topology reach, exchange-volume cap, end-to-end
  metric read from a real run log).
- `notebooks/experiments/phase4_topology.ipynb` — the DoD: three topologies on the same
  personas + committed-minority comparison, homogeneity trajectory + endpoint survey,
  plotnine figures, results → `data/phase4_topology.csv`. Executed live, 0 failures.

## DoD status

| DoD clause | State |
|---|---|
| Pluggable interaction graph, swappable at runtime | ✅ 4 `Topology` classes; `sim.topology` swap; tested |
| Same persona set, different topology | ✅ identical seeds/order across 3 configs |
| **Measurably different homogeneity metric** | ✅ dom-share 1.00 (full) vs 0.55 (sparse/clustered); entropy 0.00 vs 0.50 |
| Metric defined before the run (R14) | ✅ tested `metrics.homogeneity`; ADR 0012 |
| deterministic suite green | ✅ 89 passed, 0 failures |

## What P5/P6 need

- **P5 (persona depth + validation):** `metrics.homogeneity` is the kernel to extend —
  add embedding pairwise-distance / cluster count (rest of R14), the continuous
  dashboard, and the **R16 null-model baseline** (meaningful only once personas are the
  controlled variable). The R27 action-space-adequacy check (narrow SPEAK/ABSTAIN can
  cap observable divergence) gates reading these convergence numbers as real.
- **P5/P6 (scale):** the fast full-connectivity collapse (unanimity by tick 1 at N=20)
  suggests convergence dynamics will be sharper at N=100 — re-check trajectory shape and
  whether 4 ticks still resolves it.

## Debt / notes

- **`fully_connected` collapses very fast** (tick 1) at these seed strengths — a strong
  R10 signal, but leaves little trajectory to inspect. If P5 wants a slower, more
  legible convergence curve, lower heard-memory importance (`game_master.HEARD_IMPORTANCE`)
  or cap `exchange_volume`.
- **Committed direction is counter-intuitive** (full connectivity *reduces* a minority's
  reach by swamping it) — documented in the notebook §4/verdict so P5 doesn't mis-read it.
- **R26 tie-mutating actions not built** — event-stream seam only. The static graph
  satisfies R13's counterfactual; endogenous ties are a later dynamics feature.
- **Small-roster clamp:** `SmallWorld`/`RingLattice` clamp degree for tiny N; inert at
  the DoD's N=20, relevant only to micro tests.
