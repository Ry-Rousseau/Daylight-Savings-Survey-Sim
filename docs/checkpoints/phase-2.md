# Checkpoint — Phase 2: Game Master / interaction

**Date:** 2026-07-20 · **Status:** ✅ complete (DoD met) · **Branch:** `phase-2-game-master`

## What it proved

Two seeded agents complete resolved interactions over a tick loop, and the whole
run is durably logged with per-decision provenance. Live (`qwen/qwen3-32b`, 3 ticks,
simultaneous scheme): each agent started with 4 seed memories and ended with 4 seed
+ 3 `heard` — every SPEAK was delivered into the *listener's* private store (never
the speaker's, R2). The shared world stance-tally summed to 6 (2 agents × 3 ticks,
both spoke each tick). The evening-seeded agent spoke *permanent DST*, the
morning-seeded agent *permanent standard time* — grounded in their retrieved
memories, visible in the R29 provenance.

The SQLite run log recorded 27 events (3 `tick_marker`, 6 each of `action`,
`retrieval`, `memory_write`, `world_update`), reopened cleanly from disk (R15/R17),
and was internally consistent: one `memory_write` per delivered memory, one
`world_update` per tally increment — no orphans. Evidence:
`notebooks/experiments/phase2_interaction_dod.ipynb`.

## What's live

- `src/polis/runlog.py` — append-only SQLite `RunLog` (`runs` + `events`), closed
  event vocabulary, SHA-256 config hash (R17). ADR 0006.
- `src/polis/world.py` — tier-2 `WorldState` (roster, tick, stance tally); agents
  receive only a frozen read-only `WorldView` (R2/R3 enforced structurally). GM is
  the sole logical writer.
- `src/polis/actions.py` — closed versioned action space (`SPEAK`/`ABSTAIN`,
  `ACTION_SPACE_VERSION=1`, R23), constrained-decode json_schema, `Effect` types
  (`MemoryWrite`/`WorldUpdate`), `RetrievalProvenance`/`ActionDecision` (R29).
- `src/polis/game_master.py` — deterministic non-LLM resolution (R24): SPEAK →
  `heard` MemoryWrite per neighbour + WorldUpdate; ABSTAIN → no-op (R25); malformed
  SPEAK → no-op. ADR 0008.
- `src/polis/simulation.py` — `Population` (agents + world; `.survey` = R16 null
  baseline via the existing LangGraph fan-out), `Simulation` custom tick loop
  (R22) with the R28 update scheme (`simultaneous` default / `sequential`),
  fully-connected neighbour seam (P4 swap point), `Run` (hashed config R17;
  `metrics` an explicit stub for the P5 divergence metric). ADR 0007.
- `src/polis/memory.py` — `score_components()` (per-memory recency/importance/
  relevance/total, R29) with `score()` refactored onto it (behaviour unchanged);
  `retrieve_scored()`; `KIND_HEARD`.
- `src/polis/agent.py` — `act(topic, stances, world_view, now) -> ActionDecision`
  (retrieve → provenance → constrained action decode). `answer()` untouched.
- `src/polis/llm.py` — `decide()` for action decoding; `choose()` and `decide()`
  share one private json_schema helper, `choose()`'s observable contract unchanged
  (guarded by the live test).
- Tests: **33 deterministic passing** (13 P0/P1 + 20 new: runlog, world,
  game_master, simulation) + the live `choose()` test = 34 total. `python -m polis`
  smoke path intact.

## Observed (sanity only — not a result)

- **R28 is behaviourally real:** under `sequential`, agent 2's retrieval provenance
  includes agent 1's same-tick utterance; under `simultaneous` it does not
  (`tests/test_simulation.py`). This is the within-tick contagion knob working.
- **`choose()` contract held** — `decide()` was added by extracting a shared
  helper, not by editing `choose`'s behaviour; live test passes.

## What P3/P4 need

- **P3 (scheduling/throughput):** the tick loop is single-threaded and calls the
  endpoint once per agent per tick — batching/warm-cold (R5/R6) is where this goes
  next. The run log's serial-writer assumption holds only while the *log* writer
  stays single-threaded; revisit if P3 parallelizes logging.
- **P4 (topology):** the `topology` neighbour seam (`fully_connected`) is the swap
  point for R4/R10–R13; topology-mutating actions (R26) get their own event stream
  then. `WorldState.stance_tally` exists but agents don't yet *read* it — turning
  it into a live consensus pressure is a deliberate P4 dynamics choice.

## Debt / notes

- **R27 action-space adequacy:** the 2-action space is deliberately narrow and must
  be re-checked (separately from any homogeneity metric) before convergence results
  are read at P5 — a narrow action space can suppress observable divergence the R16
  null baseline won't catch (ADR 0008).
- **Speaker self-memory:** a speaker doesn't record having spoken; add an R19-style
  self-writeback later if cross-tick speaker coherence needs it.
- **Divergence metric (R14/R15)** is a stub (`Run.metrics = None`) by design —
  built at P5 against this log substrate.
- OpenRouter json_schema still soft-enforced (validate+retry); hard grammar (R20)
  arrives with self-hosted vLLM at P5+ (ADR 0002).
