# ADR (Architecture Design Record) 0013: Persona depth (R7–R9) + drop LLM self-hosting

Status: accepted
Phase: 5 (unit A)

## Spike question

Phase 5's spike is *"what minimum persona content prevents identity collapse over N
ticks?"*. Unit 5A resolves the sub-questions that must be settled before that can be
measured: **what shape does a value/disposition-anchored persona take (R7), and how
is persona strength measured rather than assumed (R8)?** A project-level provider
decision taken at the same moment is recorded here too.

## Context

Through P4 a `Persona` was `id / description / temperature` and all population
diversity came from seeded *memory* over a shared, deliberately neutral "resident".
R7 warns that exactly this thinness collapses into the model's generic
"helpful-assistant" voice over a run; R8 requires persona strength be *probed*, not
assumed; R9 warns diversity is necessary but not sufficient (collective convergence
can emerge with no individual cause — Ashery et al. 2025, cited in ARCHITECTURE.md).

Two constraints shaped the unit:

- **The thin persona is also the R16 null-model baseline** (ADR 0012). Whatever
  P5 adds must leave the thin persona *exactly* as-is, or the baseline it's compared
  against drifts underneath the experiment.
- **A conviction-slider end-goal** (user): eventually a slider from pre-convinced
  actors (seeded from the X post corpus at t=0) to generic/empty actors. That is
  *seed-time* opinion injection — a sibling of the *runtime* news feed (ADR 0010),
  not a replacement. 5A must not foreclose it, but must not build it (hand-authored
  personas are the instrument for now).

Separately, ADR 0002 / CLAUDE.md carried a "P5+ self-hosted vLLM (Qwen3-32B-AWQ on
RunPod/Koyeb)" direction. The user decided at P5 start that the project will **not**
self-host any LLM.

## Options considered

**Persona schema (R7):**
1. Free-text persona blob — one richer paragraph. Simple, but unversionable at the
   field level (R17 can't attribute divergence to *which* anchor) and no clean
   thin/thick contrast.
2. **Structured `values` + `dispositions` tuples added to `Persona`, defaulting
   empty.** Field-level versioning (R17), a byte-identical empty path (preserves the
   null baseline), and the two axes R7 actually names (what you care about / how you
   hold it). Composed into the system prompt in `prompts.py`.
3. Full trait model (Big-Five scores, demographic ontology, census fields now).
   Over-built for the spike; census wiring is a P6 concern and R7 says demographics
   alone are insufficient anyway.

**Persona-strength probe (R8):**
1. Assume strength from prompt content — violates R8 outright.
2. Track the categorical survey *choice* over ticks — too coarse; voice-collapse
   happens in the prose while the discrete pick is unchanged.
3. **Embed the free-text *reason* and measure cosine drift from each agent's tick-0
   baseline + distance to the population centroid.** Captures voice, gives both an
   agent-level (R8) and a collective (R9) reading, network-free-testable geometry.

**Provider:** keep the vLLM self-host seam vs drop it.

## Decision

- `Persona` gains `values: tuple[str,...]` and `dispositions: tuple[str,...]`
  (default empty), composed into the anchoring prompt; the empty persona is
  byte-identical to P0–P4 and is pinned by a regression test. Thick cast +
  `NULL_PERSONA` in `personas_nyc.py`; anchors are versioned in the run config (R17).
- R8 drift probe in `drift.py`: `capture_baseline` / `probe_drift` over embeddings
  of the agent's free-text reason, re-asked with `remember=False` so measuring does
  not perturb the memory stream it measures. Reports `drift_from_baseline` (R8) and
  `distance_to_centroid` (R9) side by side.
- **Drop LLM self-hosting for the whole project.** Stay on OpenRouter
  `qwen/qwen3-32b`. This **amends ADR 0002** (removes its P5+ self-hosted-vLLM leg).
  The local RTX 4070 Ti keeps serving lightweight local models (BGE embeddings) only.
- The conviction slider is **not built**: it rides on the new anchors + opinion seed
  memories and is left as a documented seam.

## Why

Structured anchors are the only option that satisfies R17 attribution *and* keeps
the R16 baseline immovable (empty → identical prompt) *and* gives the future slider
a dimension to turn — one choice serving three constraints. Probing the *reason*
embedding rather than the choice is what makes R8 able to see collapse at all, since
a population can hold a fixed distribution of choices while its language homogenizes.
Dropping self-hosting removes infra cost/risk from a phase whose unknowns are persona
and metrics, not backend — OpenRouter already serves every run to date.

## Consequences

- Locks the thin persona as the null baseline; any change to `persona_system`'s
  empty path must update the pinned regression string deliberately.
- `Agent.answer` grows a `remember` flag; the default (True) preserves R19 writeback.
- 5B (validation dashboard) consumes the thick cast as its controlled variable, the
  drift readings as one dashboard input, and still owes the R27 action-space-adequacy
  check before any 5C convergence number is trusted.
- Provider is now single-track (OpenRouter). Revisit only if OpenRouter drops the
  model or cost/latency at N=100 (P6) forces a change — but *not* toward self-hosting.
- The conviction slider remains unbuilt; when built it is seed-time injection layered
  on these fields, distinct from the ADR-0010 runtime feed.

## Rules touched

Implements R7 (value/disposition anchoring), R8 (measured persona strength), R9
(collective-vs-individual readings reported together). Preserves R16 (thin persona =
null baseline), R17 (anchors versioned), R19 (writeback default unchanged). Amends
ADR 0002 (R6 provider phasing — self-host leg removed).
