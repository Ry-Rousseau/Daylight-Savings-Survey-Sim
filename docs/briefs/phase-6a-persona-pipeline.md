# Brief — Phase 6a: Persona-realism seeding pipeline

**Phase:** 6 (unit A of 2) · **Rules:** R7 (real-microdata anchoring, not model-
inferred), R2/R19 (disposition seeded into private memory at t=0), R17 (versioned
persona corpus), R29 (match + generation provenance logged) · **Layer:** Persona ·
**ADRs:** 0015 (USA pivot), 0016 (disposition pipeline).

## Context

P6 seeds ~100 personas from a nationally-representative ACS PUMS sample. The load-
bearing decision (ADR 0016): **do not leave disposition to the model** — demographics-
only forces the model into stereotype point-estimates (collapsed within-cell variance,
exaggerated between-cell separation). Instead inject *real* disposition variance from a
donor survey via **statistical matching**, then use the LLM only to *elaborate* fixed
real inputs, never to invent them.

P6a builds the **re-runnable two-stage pipeline** that turns the ACS sample into a
cached, versioned **persona corpus** the simulation loads. P6b (separate brief)
consumes that corpus for the N=100 scale run. Split so one unknown lands per gate:
P6a proves the personas, P6b proves the infra.

**Dependency:** the donor disposition survey is being sourced separately. P6a's
Stage-2 logic is built and tested against a **synthetic fixture** matching the joint-
dataset schema below; the real donor data + real LLM outputs drop in when ready.

## The pipeline (ADR 0016)

```
STAGE 1 — dataset (own engineering notebook, re-runnable)
  ACS sample (data/acs_pums_sample_n100.csv, already delivered)
  + NN disposition match to donor survey(s)   [1-NN or distance-weighted
                                                sampled — never k-averaging]
  -> data/persona_seed_dataset_n100.csv        (joint: demographics + disposition items)

STAGE 2 — persona pipeline (src-first, notebook orchestrates)
  per row:
    1. backstory      LLM, constrained by row + disposition items (enact, don't invent)
    2. reflection     LLM (psychologist/economist lens), synthesizes temperament/
       -> anchor      risk/values FROM THE REAL ITEMS (not the backstory), ~2 lines
    3. assemble       reflection -> Persona.values/dispositions (always-on R7 anchor)
                      backstory + demographic facts + first-person disposition
                      statements -> seeded memories @ t=0 (importance@creation,
                      recency@sim-start); provenance logged (R17/R29)
  -> data/personas_corpus_n100.json            (the versioned persona corpus)

SIMULATION
  Population.from_corpus(...)  -> free, deterministic; no seed-time LLM calls at run time
```

**Placement is hybrid:** reflection → always-on anchor (resists R7 generic-voice
collapse); backstory/facts/disposition-statements → retrievable memory (opinion stays
R29-traceable).

**Multi-donor caveat (bake into the pipeline docstring):** fusing several donor surveys
via demographic grouping is supported, but each donor adds a conditional-independence
assumption, and items from *different* donors are never jointly observed — so a
persona's cross-donor item pairs (e.g. wake-time × extraversion) carry only
demographically-induced correlation, not empirical. Pick rich shared matching keys;
don't over-read cross-item correlations.

## Joint-dataset schema contract (Stage 1 output → Stage 2 input)

This is the **target for the disposition-sourcing work.** Stage 2 (`from_corpus`
pipeline) codes against it; tolerant to extra columns and to a configurable name map.

| Group | Columns | Notes |
|---|---|---|
| identity | `person_id` | carried from the ACS sample; stable persona id source |
| demographics | `sex, age, race_ethnicity, hispanic_origin, education, occupation, industry, marital_status, presence_of_children, family_size, region/state/puma` | as delivered in `acs_pums_sample_n100.csv` — the matching keys |
| disposition items | e.g. `bf_openness … bf_neuroticism` (percentile/score), `value_*` (e.g. Schwartz), `attitude_*` (e.g. trust/routine), optional `chronotype`/`wake_time` | **borrowed from the donor survey via NN match**; any subset present is used |
| match provenance | `donor_respondent_id`, `match_distance` | which real respondent each persona matched to + match quality (R29) |

Requirements on the donor survey: it must carry disposition-adjacent items **and**
demographic fields overlapping the ACS matching keys (age, sex, race, education,
region at minimum; occupation/marital/children a bonus). Disposition columns are all
optional to Stage 2 — whatever is present flows into the backstory + reflection inputs.

## Definition of done

- `src/polis/persona_pipeline.py` (new): `SeededPersona` (a `Persona` + its t=0 seed
  memories + provenance); `build_persona(row, client, *, seed)` running backstory →
  reflection → assemble; `build_corpus(df, client, *, seed)` over a joint dataframe;
  JSON serialize/deserialize of the corpus.
- Prompt templates (in `prompts.py`): `backstory_user(row, disposition_items)` and
  `reflection_user(row, disposition_items)` (Variant A — from items). Deterministic,
  reviewable strings.
- Model-free helpers: row → factual demographic description; disposition items →
  first-person statements ("I'm up before six most days"; "I don't chase reinvention").
- `Population.from_corpus(corpus_or_path, *, client, embedder=…)` builds the
  population from the cached artifact; anchor → system prompt, seed memories → each
  agent's private store at t=0.
- `notebooks/engineering/persona_pipeline.ipynb` (or `.qmd`) — orchestrates Stage 2
  end-to-end, re-runnable, writing the corpus artifact; validated on a small n before
  the full 100.
- Corpus is **R17-versioned** (seed + content hash + generating model/prompt) and the
  simulation run config cites it.
- Deterministic suite green (existing 132 + new pipeline tests) using a **stub client**
  for the LLM stages; P0–P5 behaviour unchanged.

## Prerequisites

Phases 0–5 green. New branch `phase-6a-persona-pipeline`. `data/acs_pums_sample_n100.csv`
delivered. Donor disposition survey **pending** (build against a synthetic fixture
until it lands). `.venv` (3.11); BGE embeddings local (seed-memory embedding).

## Ordered tasks

1. Synthetic fixture: a small joint-dataset CSV/dataframe matching the schema
   (a few rows, plausible disposition items) for TDD. `tests/fixtures/`.
2. `src/polis/persona_pipeline.py` — `SeededPersona`, model-free helpers (row →
   description, items → first-person statements), corpus JSON (de)serialize. TDD.
3. `prompts.py` — `backstory_user`, `reflection_user` (Variant A). Tests assert the
   fixed inputs appear and the reflection prompt reads items, not backstory.
4. `build_persona` / `build_corpus` with a **stub client** (deterministic canned
   backstory/reflection) → assert anchor populated from reflection, memories carry
   backstory + facts + disposition statements, provenance recorded. TDD.
5. `Population.from_corpus` (in `simulation.py` or `persona_pipeline.py`) + R17 config
   cites the corpus id/hash. TDD a round-trip: corpus → population → personas intact.
6. `notebooks/engineering/persona_pipeline.ipynb` — orchestrator; a dry run on the
   synthetic fixture (real donor data + live LLM validated when the dataset lands).
7. Flip ADR 0016 to accepted on review; hand off to P6b.

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest -q` green (132 + new).
- Round-trip: `build_corpus(fixture, stub)` → JSON → `from_corpus` → a `Population`
  whose personas carry the reflection anchor + the seeded t=0 memories + provenance.
- Reflection prompt is built from the disposition **items**, not the backstory
  (asserted) — the ADR-0016 Variant-A guarantee.
- A missing disposition column degrades gracefully (backstory/reflection use what's
  present); a persona with no disposition items still builds (falls back toward the
  demographic-only description).
- Corpus artifact carries a version/hash + generating model/prompt; two builds with
  the same seed + stub are byte-identical (determinism).

## Hand-off pointer

P6b inherits `Population.from_corpus` + the versioned corpus artifact as its
population source, and runs the 3-arm N=100 scale contrast (null / demographic-only /
full-pipeline) on it. When the donor dataset lands: wire Stage 1's NN-match, run the
pipeline live on n=100, commit the corpus artifact, then P6b's live run + spend
approval.
