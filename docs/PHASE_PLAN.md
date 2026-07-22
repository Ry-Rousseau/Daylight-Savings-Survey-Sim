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
| **8 — Interactive GUI** *(optional / stretch)* | Can a thin front-end drive the already-parameterised engine — set N, topology, tick count, opinionated/feed fraction → run → watch the divergence metric — without leaking logic upward into the UI? | A small local app (e.g. Streamlit/Gradio) with sliders bound to the existing config fields (R1/R4/R12/R3) and a live divergence-trajectory plot; **no new engine logic** — the UI only *calls* `src/polis` | A non-technical user runs a simulation end-to-end from sliders and sees the divergence trajectory; every knob maps 1:1 to a logged run-config field (R17) | Interface / Query | (none new — consumes R14–R17) |

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
- Phase 8 (GUI) is optional and additive: it introduces **no engine logic**, only
  a front-end that drives the existing config fields, so it can slot in any time
  after P5 (which gives it a divergence metric to display). Cross-cutting seams it
  would expose — the external-signal **feed** (ADR 0010) and the topology/dynamics
  knobs — are built with the engine, not with the UI.

## Status

Update this section as phases complete.

- [x] Phase 0 — walking skeleton (2026-07-19); see checkpoints/phase-0.md
- [x] Phase 1 — memory (2026-07-20); DoD met, P(permanent-DST) delta +1.000; see checkpoints/phase-1.md. Model-capability sweep addendum done (ADR 0005): baseline revised to Qwen3-32B; 8B's pass depended on annotated options, 14B/32B don't.
- [x] Phase 2 — Game Master / interaction (2026-07-20); DoD met, 2 agents complete resolved SPEAK interactions over a tick loop, world + memory update consistently, durable SQLite run log with R29 provenance reopens from disk; see checkpoints/phase-2.md. ADRs: 0006 (SQLite run log), 0007 (simultaneous update default, R28), 0008 (SPEAK/ABSTAIN action space v1).
- [x] Phase 3 — Scheduling & throughput (2026-07-20); DoD met. Concurrent per-agent-per-tick scheduler (R5) + latency/token/cost logging (R6); live sweep of 25 agents × 3 ticks over concurrency [1,4,8,16] ran with **0 failures**, throughput scaling 5.7× (0.24→1.36 decides/s), latency ~flat — endpoint concurrency-bound. Premise reframed (ADR 0009): local GPU batching of the 32B model is infeasible on the 16 GB card and premature per ADR 0002, so P3 benchmarks request concurrency against managed OpenRouter and the vLLM continuous-batching path is a seam benchmarked at P5. 53 deterministic tests green. See checkpoints/phase-3.md. ADR 0009.
- [x] Phase 4 — Topology (2026-07-20); DoD met. Topology is a pluggable, seeded, swappable parameter (`src/polis/topology.py`: fully-connected / ring / small-world / stochastic-block); the P4-scoped homogeneity metric (`src/polis/metrics.py`) is defined+tested before the run (ADR 0012). Live run (20 agents × 4 ticks, same personas, 0 failures): `fully_connected` collapses to unanimity by tick 1 (dominant share 1.00 / entropy 0.00) while `small_world`/`stochastic_block` hold the split (0.55 / 0.50) — R10 confirmed. Committed minority (R11): a 4-agent faction persuades 0% under full connectivity vs 31% under clustering (full exposure swamps it; clustering shelters it). Exchange volume (R12) + committed-minority affordance (R11) shipped; R26 tie-mutation reserved as a seam. 89 tests green. See checkpoints/phase-4.md. ADRs 0011 (topology), 0012 (metric boundary).
- [x] Phase 5 — Persona depth + validation wiring (2026-07-20); DoD met. Split into 5A (persona depth, R7–R9), 5B (validation dashboard, R14–R17 + R27), 5C (live DoD run). `Persona` gains value/disposition anchors (`personas_nyc.py` thick NYC cast + `NULL_PERSONA`); R8 drift probe (`drift.py`); full R14 layer on the P4 kernel (`metrics.py`: embedding pairwise-dispersion + cluster count, `divergence_trajectory`/`divergence_summary`, `action_space_adequacy`). Live run (12 agents × 5 ticks, thick vs null baseline, 0 failures): **both arms converge on stance (dom-share→1.00) but the embedding axis splits them** — thick holds 8 voice-clusters / dispersion 0.20 & drift 0.127, null collapses to 1 cluster / 0.12 & drift 0.191. Persona depth protects the *voice*, not the *vote* (R7 resists voice collapse; R9 — stance consensus still emerges). R27 gate passed (no flags). 132 tests green. See checkpoints/phase-5.md. ADRs 0013 (persona schema + drop self-hosting, amends 0002), 0014 (validation dashboard). **No vLLM — self-hosting dropped project-wide.**
- [x] Phase 6 — Scale to N=100 (2026-07-22); DoD met (100-agent runs, clean logs, R5/R6/R17). Went well beyond scale: NYC→USA pivot (ADR 0015), real-microdata disposition persona pipeline (ADR 0016), opinion/conviction layer, action-space maturity (SHARE_CONSIDERATION/REBUT + deliberate mode, ADR 0017/0018), and the **YouGov calibration instrument**. **Headline finding:** base LLMs express the *expert* DST consensus (permanent standard) not the *public* preference (permanent DST) — Qwen inverts reality, **Claude Sonnet-5 calibrates near YouGov (45% DST ≈ 50%)**, so the individual-level collapse was a model artifact; residual vote-convergence is genuine R10 dynamics. **Open realism thread:** demographic gradient still mis-calibrated (inverted age trend). See `docs/checkpoints/phase-6.md` (the realism struggle is the pickup point).
- [ ] Phase 7
- [ ] Phase 8 — Interactive GUI *(optional / stretch)*: a thin sliders-and-run front-end over the already-parameterised engine, added after P5's divergence metric exists to have something meaningful to display. UI calls `src/polis`, never redefines it. Sibling seam: the external-signal **feed** (ADR 0010) is one of the knobs such a UI would expose.