# Phase Plan

## Principle

This project is built as a walking skeleton, thickened one axis of
complexity at a time. Every phase boundary is a runnable checkpoint that
isolates exactly one new unknown, so that if something breaks, the phase
that caused it is unambiguous. Full ambition (all five layers, full rule
set, 100 agents, mature survey subsystem) is the target of the *whole* plan,
not of any single phase.

Each phase should end with:
- A **spike question** resolved and written up as an ADR in `docs/adr/`
- A **definition of done** that is objectively checkable, not vibes-based
- A **working system** — nothing left half-wired

## Two rule buckets

Not every rule in `ARCHITECTURE.md` needs code written for it in the phase
where it's first listed. Split rules into:

- **Build requirements** — need actual implementation this phase
- **Standing constraints** — need to simply not be violated as later features
  are added (cost nothing until convenience tempts you to break them)

---

## Phase table

| Phase | Spike question(s) | Deliverable | Definition of done | Layers touched | Rules activated |
|---|---|---|---|---|---|
| **0 — Walking skeleton** | Does local model + constrained decoding reliably return schema-valid JSON? Does a minimal LangGraph fan-out/gather work for 3 agents? | 3 hardcoded agents, no memory, one hardcoded survey question, structured output only | 3 agents answer 1 question, 3 parseable responses returned | Architecture (minimal) | R1, R20 |
| **1 — Memory** | Which local embedding model is fast enough at this scale? Does recency/importance/relevance retrieval actually change output? | In-memory or SQLite vector store; recency/importance/relevance scoring; seeded memories per agent | Two agents with different seeded memories give measurably different survey answers | Architecture, Persona (begins) | R2, R19 |
| **2 — Game Master / interaction** | Minimal action schema + resolution logic for one interaction type? Simultaneous vs sequential within-tick update (R28)? | Symbolic action-resolution layer; world-state store separate from agent memory; **`Simulation`/`Population` container + custom tick loop**; **durable run-log substrate** (foundation for R15/R17) with **decision provenance** logged (R29) | Two agents complete one resolved interaction; both memories and world state update consistently; the run log records the interaction with each decision's retrieval provenance | Architecture, Dynamics (begins) | R2, R3, R28, R29 |
| **3 — Scheduling & throughput** | What agent count / tick rate is sustainable on home hardware with batched local inference? | Warm/cold scheduler; vLLM (or Ollama) batching benchmarked | 20–30 agents run multiple ticks without failure, with logged latency/cost | Architecture | R5, R6 |
| **4 — Topology** | How much does the divergence metric change under different graph structures at this N? | Pluggable interaction graph (fully-connected vs. clustered/small-world), swappable at runtime | Same persona set, different topology, measurably different homogeneity metric | Dynamics | R4, R10–R13 |
| **5 — Persona depth + validation wiring** | What minimum persona content prevents identity collapse over N ticks? | Value/disposition-anchored personas; periodic drift probing; full R14–17 metrics logged per tick | Full validation dashboard running against a null-model baseline | Persona, Validation | R7–R9, R14–R17 |
| **6 — Scale to full population** | Does the phase-3 infra plan hold at 100 agents, or does something non-linear break? | Push N to 100; re-run infra benchmarks | Full 100-agent run completes with clean logs | Architecture (re-validated) | R5, R6, R17 |
| **7 — Survey subsystem maturity** | Do repeated surveys over simulated time produce coherent, traceable answer evolution rather than noise? | Multi-timepoint survey capability; response-to-memory writeback; optional calibration check | Same survey run twice at different simulated timepoints, differences attributable to logged events | Persona, Validation | R18, R19, R21 |

---

## Notes on sequencing

- Phases 0–2 are pure architecture: cheap to debug, no live-population cost.
- Phase 2's run-level scaffolding (the `Simulation`/`Population` container, the
  durable run-log substrate, and decision provenance) is designed ahead in
  `docs/design/run-architecture.md`; the storage-tech choice and the R28
  tick-update scheme firm into ADRs at P2. The retrieval store stays in-RAM
  numpy — persistence is additive, no global vector index (R2).
- Phase 3 is infrastructure/throughput: validate before scaling agent count,
  not after.
- Phases 4–5 are where the actual research questions live (dynamics and
  persona) — deliberately sequenced *after* infra is proven solid, so a bad
  result there is a finding, not a bug.
- Phases 6–7 are scale-out and the actual deliverable (survey the
  population). Nothing new architecturally should be discovered here if
  0–5 were done properly — this is where that assumption gets tested.

## Status

Update this section as phases complete.

- [x] Phase 0 — walking skeleton (2026-07-19); see checkpoints/phase-0.md
- [x] Phase 1 — memory (2026-07-20); DoD met, P(permanent-DST) delta +1.000; see checkpoints/phase-1.md. Model-capability sweep addendum done (ADR 0005): baseline revised to Qwen3-32B; 8B's pass depended on annotated options, 14B/32B don't.
- [x] Phase 2 — Game Master / interaction (2026-07-20); DoD met, 2 agents complete resolved SPEAK interactions over a tick loop, world + memory update consistently, durable SQLite run log with R29 provenance reopens from disk; see checkpoints/phase-2.md. ADRs: 0006 (SQLite run log), 0007 (simultaneous update default, R28), 0008 (SPEAK/ABSTAIN action space v1).
- [ ] Phase 3
- [ ] Phase 4
- [ ] Phase 5
- [ ] Phase 6
- [ ] Phase 7