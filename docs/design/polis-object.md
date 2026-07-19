# The object — `polis`

## What it is
`polis` is a configurable **opinion-simulation engine**. It seeds a population of ~100 LLM personas from NYC census data and lets us **survey** them with single-select questions (headline case: daylight saving time) to estimate the real population's opinion. It is the deliverable; notebooks import *down* from it and never redefine its logic (see `conventions.md`).

## Narrow public interface (sketch — firms up in P1)

```python
Population.from_census(spec, *, seed) -> Population
    # per-agent config (temperature / top_p / model, topology slot) are FIELDS, not globals (R1/R2)

population.survey(question) -> AnswerDistribution
    # single-select -> distribution over options; the headline op.
    # Surveying with no dynamics run = the R16 null-model baseline.

sim = Simulation(population, topology, dynamics_cfg)
sim.run(ticks) -> Run
    # per-tick logged trajectory + divergence metric (R14/R15)

run.config    # architecture params + persona set + topology, hashed (R17)
run.metrics   # divergence trajectory: pairwise distance / dominant-share / cluster count
```

Everything hangs off two headline operations: **seed a population** and **survey it** (optionally after running interaction dynamics).

## Layer map (where the R-rules bite)
- **L1 Engine:** per-agent generation params; strict global (shared, read-only) vs per-agent private state; model pinned & logged; no cross-agent sharing of cached reasoning (R1–R6). Backend: local vLLM / `Qwen3-8B-AWQ`.
- **L2 Persona:** value/disposition-anchored, strength measured not assumed, diversity necessary-not-sufficient (R7–R9).
- **L3 Dynamics:** pluggable/freezable topology, tunable exchange volume, committed-minority affordance (R10–R13).
- **L4 Validation:** divergence metric defined *before* runs, logged per tick, against a null baseline, versioned by config (R14–R17).

## Subsystems under the prompt (`requirements_features.md`)
Embedding-based **memory retrieval** (recency / importance / relevance + reflection) · **constrained/structured output** → a fixed action vocabulary · a **non-LLM world-state / action-resolution** layer ("Game Master" pattern). Sequenced per the phase plan: structured output lands in the Phase 0 walking skeleton, memory in Phase 1, the Game Master in Phase 2, topology in Phase 4, persona depth + full validation in Phase 5.
