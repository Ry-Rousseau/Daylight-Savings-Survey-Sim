# Brief — Phase 2: Game Master / interaction

**Branch:** `phase-2-game-master` · **Decisions:** `docs/adr/0006-runlog-storage.md`,
`0007-tick-update-scheme.md`, `0008-action-space-v1.md`

## Context

Phases 0–1 built the per-agent spine: thin personas answer one DST survey through the
unchanged `LLMClient.choose()`, each with a private numpy `MemoryStore` (retrieve → inject →
answer → R19 writeback). Everything is ephemeral RAM and operates only at the `Agent` level.

Phase 2 thickens exactly one axis — **interaction** — and introduces the run-level scaffolding
the R-rules already assume but no phase owns yet (designed ahead in
`docs/design/run-architecture.md`): a symbolic **Game Master**, a **world-state store**
separate from agent memory, the **`Simulation`/`Population` container + custom tick loop**, and
a **durable run-log substrate** with **decision provenance**. Rules activated: R2, R3, R23,
R24, R25, R28, R29 (foundations for R15/R17); R22's custom-tick-loop half lands here.

Confirmed decisions: run-log storage = **SQLite**; default within-tick update scheme =
**simultaneous**; interaction shape = **SPEAK + ABSTAIN, one-directional**.

### Deliberately NOT in P2 (walking-skeleton discipline)
Divergence **metric** (R14/R15) → P5 · pluggable **topology** (R4, R10–R13) → P4 · census
seeding (`from_census`) → P5 · **scheduling/batching** (R5/R6) → P3 · **persona depth**
(R7–R9) → P5 · reflection + hard grammar decoding (R20) → P5.

## Definition of done

A 2-agent `Simulation` runs ≥1 tick and, for a resolved SPEAK:
- the speaker's stance appears as a new `kind=heard` memory in the **listener's** store (never
  the speaker's), and the world stance-tally increments for that stance;
- both agents' memory + the world tally change **consistently** with the logged effects — every
  memory-write / world-update has a corresponding log event, no orphans;
- the **SQLite run log** records, for that tick: the actions, the effects, and an **R29
  retrieval-provenance** event per decision (the scored memory set with recency/importance/
  relevance components + total);
- reloading the log from disk in a fresh connection reproduces the event stream (R15/R17);
- deterministic no-network pytest covers GM resolution, the R28 update scheme (sequential lets
  agent 2 hear agent 1 same-tick; simultaneous does not), the world/agent write boundary, and
  run-log round-trip;
- a DoD notebook demonstrates it live (2 agents, a few ticks) and shows the log + provenance.

`LLMClient.choose()`'s observable contract stays unchanged (existing live test is the guard).

## Prerequisites

- On branch `phase-2-game-master` off `main` (done).
- P0/P1 tests green; `choose()` contract intact; BGE-small embeddings on CUDA available.

## Ordered tasks

1. **`src/polis/runlog.py`** — SQLite append-only `RunLog`: `open(path)`, `log_run(config) ->
   run_id` (stores `config_json` + `config_hash`, R17), `log_event(run_id, tick, agent_id,
   event_type, payload)`, `events(run_id)`. Event types: `tick_marker`, `action`,
   `memory_write`, `world_update`, `retrieval`. + `tests/test_runlog.py` (round-trip, hash).
2. **`src/polis/world.py`** — `WorldState` (roster, tick, `stance_tally`), read-only view to
   agents, GM-sole-writer mutations (R2/R3). + `tests/test_world.py` (write boundary).
3. **`src/polis/actions.py`** — `ActionType`(SPEAK/ABSTAIN), `Action`, `ACTION_SPACE_VERSION`,
   the constrained-decode json_schema, `RetrievalProvenance`, `Effect` types (`MemoryWrite`,
   `WorldUpdate`).
4. **`src/polis/game_master.py`** — `GameMaster.resolve(action, actor_id, world_view,
   neighbors) -> list[Effect]`, deterministic (R24): SPEAK → `MemoryWrite`(kind=heard) per
   neighbor + `WorldUpdate`; ABSTAIN → `[]` (R25); malformed SPEAK → ABSTAIN. +
   `tests/test_game_master.py`.
5. **`memory.py`** — add `score_components()` (per-memory recency/importance/relevance/total for
   R29; `score()` refactored onto it, behavior unchanged) + `KIND_HEARD`.
6. **`agent.py`** — `act(world_view, *, now) -> ActionDecision` (retrieve → capture provenance →
   constrained action decode). `answer()` untouched.
7. **`llm.py`** — extract the json_schema call+validate+retry core into a private helper;
   `choose()` delegates (no observable change); add `decide(system, user, schema, temperature)`
   for the action schema. **`prompts.py`** — action-decision prompt.
8. **`src/polis/simulation.py`** — `Population` (agents + `WorldState`; `.survey` delegates to
   `run_survey`), `Simulation(population, topology, dynamics_cfg, logger)` custom tick loop
   (R22) with R28 update scheme, neighbor seam (fully-connected default), `Run` (hashed config
   R17; `metrics` explicit stub). + `tests/test_simulation.py` (both schemes, consistency).
9. **`__init__.py`** — export the new public objects.
10. **DoD notebook** `notebooks/experiments/phase2_interaction_dod.ipynb` (live).
11. **ADRs 0006–0008**, then update `status.md`, tick Phase 2 in `PHASE_PLAN.md`, write
    `docs/checkpoints/phase-2.md`.

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest` green — new deterministic tests + P0/P1 unchanged.
- SPEAK writes land in the **listener's** store only; GM is the sole writer of world + cross-
  agent memory (grep/structure check, R2/R3).
- Run log round-trips from a reopened DB file; each decision has an R29 `retrieval` event.
- `choose()` diff shows no observable-contract change; existing live test passes.
- `.venv/Scripts/python.exe -m polis` still works (P0 smoke path intact).

## Hand-off pointer

On completion: update `status.md`, tick Phase 2 in `PHASE_PLAN.md`, write
`docs/checkpoints/phase-2.md` (what it proved, what's live, what P3/P4 needs). Commits are the
user's to approve.
