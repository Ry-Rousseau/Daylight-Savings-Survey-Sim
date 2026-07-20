# Brief — Phase 5A: Persona depth

**Phase:** 5 (unit A of 3) · **Rules activated:** R7 (value/disposition anchoring), R8 (persona strength measured, not assumed), R9 (diversity necessary-not-sufficient) · **Layer:** Persona.

## Context

Phase 5 is two axes at once (Persona + Validation), too big for one unit. It is
cut into three: **5A persona depth** (this brief), **5B validation dashboard**
(embedding pairwise-distance / cluster count / continuous dashboard / R16 null
baseline / R27 adequacy), **5C live DoD run + checkpoint** (thin vs thick × drift
trajectory × dashboard vs null). 5A must land first because the **R16 null-model
baseline is only meaningful once personas are the controlled variable** (ADR 0012).

Through P4 the population's diversity came entirely from **seeded memory** over a
*shared, deliberately neutral* `Persona("resident", …)` — the persona object itself
is thin (`id / description / temperature`). R7 says exactly this thinness collapses
into the model's generic "helpful assistant" voice over N ticks. So the persona
identity axis is genuinely unbuilt, and its spike is: **what minimum persona content
prevents identity collapse over N ticks?**

**Two project decisions taken at P5 start (steer given):**
- **Hand-authored value/disposition personas** are the spike instrument this phase.
  ACS PUMS census→persona wiring stays a documented seam, finalized at P6 when we
  actually need 100 demographically-realistic agents (R7 says demographic labels
  *alone* are insufficient anyway, so census is not what the spike turns on).
- **No LLM self-hosting, ever** — stay on OpenRouter `qwen/qwen3-32b` for the whole
  project. The ADR-0002 "P5+ self-hosted vLLM" direction is dropped (ADR 0013
  amends 0002). This removes vLLM from P5's scope entirely.

**Conviction-slider seam (do not build, do not foreclose).** The end-goal is a
slider from **pre-convinced actors** (seeded at t=0 from the X post corpus) to
**generic/empty actors**. That slider is *seed-time opinion injection*, a sibling of
the *runtime* news feed (ADR 0010) — not the same thing, and it does not supersede
it (they are two delivery channels for the same corpus: seed-time vs per-tick). 5A's
job is only to make the persona schema rich enough that a conviction axis has
something to turn later (values/dispositions + composable opinion seeds), not to
build the slider. The thin persona (empty values/dispositions) is preserved exactly
as-is, which makes it double as the **R16 null-model persona** for 5B/5C.

## Definition of done

- `Persona` carries **value/disposition** content (R7), backward-compatible: an
  empty persona composes the *identical* prompt it does today, so the thin persona
  survives unchanged as the null baseline. `system_prompt()` + `prompts.persona_system`
  fold values/dispositions into the anchoring prompt.
- A hand-authored **cast of thick NYC personas** (value/disposition-anchored, with
  matching opinion seed memories where relevant) exists as the 5B/5C spike instrument.
- An **R8 persona-strength / drift probe** (`src/polis/drift.py`): capture each
  agent's tick-0 baseline voice, then measure per-probe **drift-from-own-baseline**
  and **distance-to-population-centroid** (the R9 collective-collapse signal), from
  embeddings — pure math network-free-testable.
- Personas are **versioned in the run config** (R17) with their values/dispositions,
  so a thick-vs-thin run's outcome is traceable to persona content, not just id.
- Deterministic suite green (existing 89 + new persona/drift tests); the P0–P4
  behavior is unchanged (thin persona prompt byte-identical).

## Prerequisites

Phases 0–4 merged to `main`. Branch `phase-5a-persona-depth` (off `main`). `.venv`
(Python 3.11). BGE-small embeddings already local (used by the drift probe).

## Ordered tasks

1. `src/polis/persona.py` — `Persona` gains `values: tuple[str,...]` +
   `dispositions: tuple[str,...]` (default empty); `system_prompt()` passes them
   through. TDD `tests/test_persona.py` (composition + thin-persona byte-identity).
2. `src/polis/prompts.py` — `persona_system(description, values=(), dispositions=())`
   composes the anchoring prompt; empty → today's exact string.
3. `src/polis/personas_nyc.py` — hand-authored thick cast (value/disposition +
   opinion-leaning seed specs reusing the `memory_seeds` shape), plus the thin
   `NULL_PERSONA` re-exported for the baseline.
4. `src/polis/drift.py` — pure: `cosine_distance`, `population_centroid`,
   `centroid_distance`; orchestration: `capture_baseline(agents, probe)` /
   `probe_drift(agents, probe, baselines)` over `Agent.answer` (embeds the free-text
   *reason*, where voice-collapse shows). TDD `tests/test_drift.py` on synthetic vecs.
5. `src/polis/simulation.py` — `_build_config` personas block includes
   `values`/`dispositions` (R17).
6. `docs/adr/0013-*.md` — persona schema + drift instrument + vLLM-drop (amends 0002).

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest -q` green.
- `test_persona.py` proves: empty persona prompt == the P0–P4 string (regression
  guard for the null baseline); a thick persona's prompt contains its values +
  dispositions.
- `test_drift.py` proves: zero drift for identical vectors, max for opposite;
  centroid distance shrinks as a population converges (the R9 signal).
- Run config for a thick population carries values/dispositions per persona.

## Hand-off pointer

Update `status.md` (5A done, 5B next), leave `PHASE_PLAN.md` P5 open. 5B inherits:
the thick cast + `NULL_PERSONA` (its controlled variable + null baseline), the drift
probe (one input to the dashboard), and the R27 adequacy check still owed before any
convergence number from 5C is trusted.
