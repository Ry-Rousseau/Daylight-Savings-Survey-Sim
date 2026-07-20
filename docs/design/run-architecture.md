# Run-level architecture — state tiers, the Simulation container, and provenance

**Status:** forward-looking design note (authored during Phase 1 close, before
Phase 2). It scopes structure the R-rules already assume but no phase yet owns.
Firm decisions here become ADRs at their phase (storage tech + tick model at P2).
Complements `polis-object.md` (which describes the *interface*); this describes
the *run-level scaffolding* behind it.

## Why this note exists

Phase 1 built a sound per-agent spine (persona + private memory + query). But
three things sit between here and the full vision (100 agents, tick loop,
topology, per-tick validation, surveys over simulated time) that are currently
un-homed:

1. **Persistence / logging** — R15 wants metrics logged *per tick*, R17 wants
   every run versioned against its config so divergence is traceable, P7 wants
   surveys *over simulated time*. All assume durable state. Today everything is
   ephemeral RAM.
2. **The Simulation/Population container** — `polis-object.md` promises
   `Population.from_census(...)` and `Simulation(pop, topology).run(ticks)`, but
   we currently operate only at the `Agent` level. The container that owns the
   agents + clock + topology + logger is diffuse across P2/P4/P6 with no design.
3. **Interpretability / provenance** — knowing *why* an agent answered. We have
   the model's self-reported `reason`; we do not yet record the retrieved-memory
   set + scores that actually conditioned the answer.

## Three state tiers (the R2/R3 boundary, made concrete)

| Tier | What | Lifetime | Access | Rule |
|---|---|---|---|---|
| **Private agent state** | Per-agent `MemoryStore` (memories, beliefs) | Per agent, across ticks | Owned by one agent; never shared | **R2** — structural, one store per agent |
| **Shared world state** | Environment / Game-Master world model; any deliberately shared signal (news feed) | Per run | Read-only to agents; written only by the Game Master | **R2/R3** — separate store; shared signals logged |
| **Durable run log** | Append-only event stream: memory writes, retrievals (+scores), actions, survey responses, per-tick metrics | Persisted beyond the process | Write-only during the run; queried offline for validation/interpretability | **R15/R17** — per-tick, config-versioned |

The current code implements tier 1 only. Tier 2 arrives with the Game Master
(P2). Tier 3 is the substrate this note argues to add — and it is *additive*:
it does not change tiers 1–2, it observes them.

## Where memory lives (answering "is RAM permanent?")

**No — but the retrieval store stays in-RAM numpy; persistence is a layer under
it, not a replacement of it.**

- **Hot path:** brute-force numpy cosine over each agent's own vectors is the
  right tool well past the 100-agent target (thousands of vectors/agent =
  sub-ms). ADR 0004 revisits this *only* on a scale trigger (~P6), and even then
  toward a per-agent-namespaced store, never a global `agent_id`-filtered index
  (that regresses R2 to separation-by-convention).
- **Persistence:** each agent's stream serializes (`MemoryStore.to_list()`
  exists) into the tier-3 run log — for cross-run survival, replay, and
  analysis. A vector DB is *not* required for retrieval speed; the actual need is
  durability + analytical queryability, which the run log provides.
- **Net:** keep numpy for retrieval; add durable logging beside it. Do not swap
  the retrieval mechanism for a DB on performance grounds.

## The Simulation / Population container (P2+)

A single object ties the tiers together and hosts the **custom tick loop** —
the non-LangGraph half of R22 (LangGraph stays scoped to the bounded survey
fan-out; it does not run the simulation core).

```
Population        # owns the agents (private state, tier 1) + the shared world state (tier 2)
  .from_census(spec, *, seed) -> Population
  .survey(question) -> AnswerDistribution     # tier-1 read; the R16 null baseline when no ticks run

Simulation(population, topology, dynamics_cfg, logger)
  .run(ticks) -> Run
    # per tick: schedule agents -> each reads memory -> acts (R23/R24 via Game Master)
    #           -> world state + memories update -> logger records the tick (tier 3)
  Run.config   # architecture params + persona set + topology, hashed (R17)
  Run.metrics  # divergence trajectory, logged per tick (R14/R15)
```

Open decisions to firm into ADRs at P2:
- **Storage tech** for tier 3 — SQLite vs parquet vs jsonl (lean: SQLite for
  queryability + single-file durability; parquet if analysis is columnar-heavy).
- **Tick concurrency + read/write ordering** — within a tick, when does each
  agent read vs write memory (simultaneous vs sequential update)? This is a
  homogenization-relevant choice (simultaneous update reduces within-tick
  contagion), so it belongs on the ARCHITECTURE rule set, not left to convenience.
- **World/agent write boundary** — enforce that agents cannot write tier 2.

## Interpretability / provenance

Two distinct signals, only one of which we have:

- **Self-reported rationale (have it):** `choose()` → `SurveyAnswer.reason`. Soft
  — Phase 1 showed the model regurgitates injected memories, so `reason` is a
  post-hoc story, useful but not mechanism.
- **Retrieval provenance (to build):** the *scored memory set* actually surfaced
  for the decision — which memories, with what recency/importance/relevance
  components and final score. This is the mechanistic answer to "why did agent X
  say Y," and the raw material for tracing how a memory propagates into
  convergence.

Cheap to add: `MemoryStore.score()` already computes the per-memory components
and `retrieve()` already returns the hits — provenance is exposing that scored
set from `Agent.answer()` and writing it to the tier-3 log per decision. Build it
with the logging layer at P2; do not half-wire it into the P1 close.

## Summary of what to scope

- **P2:** tier-3 run log (SQLite lean) + world-state store (tier 2); the
  `Simulation`/`Population` container + custom tick loop; retrieval provenance
  logged per decision; tick read/write-ordering as a new ARCHITECTURE rule.
- **Standing:** retrieval store stays numpy; persistence is additive; no global
  vector index (R2).
