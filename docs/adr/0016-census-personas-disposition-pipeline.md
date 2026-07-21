# ADR (Architecture Design Record) 0016: Census personas via real-microdata disposition matching + LLM elaboration

Status: proposed
Phase: 6

## Spike question

**What goes on a persona beyond its census demographics, such that the population is
diverse and realistic without the persona layer becoming a bias source?** And
downstream: does injecting *real* disposition data produce more realistic (less
stereotyped) diversity than either demographics-only or model-generated dispositions?

## Context

P6 seeds ~100 personas from a nationally-representative ACS PUMS sample (delivered by
a separate pipeline: `notebooks/engineering/acs_pums_national_sample.qmd` →
`data/acs_pums_sample_n100.csv`, columns: `sex, age, race_ethnicity, hispanic_origin,
education, occupation, industry, marital_status, household_relationship,
presence_of_children, family_size, region/state/puma`). Demographics are real; the
open question is everything *beyond* them.

An earlier draft of this ADR chose **demographics-only** (compose a factual
description, leave `values`/`dispositions` empty, treat the model's latent
demographic→opinion map as the estimator). That was **reversed** for a specific
reason:

- **Demographics-only is not neutral — it is a stereotype amplifier.** Given only
  demographic labels, the model cannot know a synthetic person's disposition, so it
  reverts to the *conditional mode* — the stereotype centroid. This **collapses real
  within-cell variance** (every "53yo Georgia sales rep" sounds identical) and
  **exaggerates between-cell separation** (each cell becomes its cartoon). So leaving
  disposition "open" doesn't avoid bias; it hands disposition to the model's most
  stereotyped guess.

The fix is to supply *real* disposition variance from real people, model-free, so the
persona layer restores the true conditional spread instead of the model's point
estimate. This is **statistical matching / data fusion**: borrow disposition items
from a donor survey onto the ACS recipient via shared demographic keys. Its load-
bearing assumption is conditional independence (disposition ⊥ ACS-only fields given
the matched demographics) — documented as a limitation.

Note on what this buys *for DST specifically*: the DST **stake** still comes mostly
from lifestyle demographics (occupation→schedule→daylight), not personality. The
disposition layer's main job is to **de-stereotype the voice** and restore within-cell
heterogeneity — precisely the concern above. (A morningness/chronotype item, if a
donor survey carries one, is the exception — directly DST-relevant.)

## Options considered

1. **Demographics-only** (prior draft). Rejected: stereotype amplifier (above);
   collapses the within-cell variance the whole exercise needs.
2. **LLM-generated dispositions/backgrounds.** Rejected as the source: fabricates and
   is circular — the model invents the personality it then simulates, so any diversity
   is the model's own stereotype spread (the R16 artifact). Retained only as a possible
   later ablation.
3. **Social-media (X/LinkedIn) dispositions.** Rejected *as persona content*: cannot
   honestly link a scraped stranger to a synthetic census draw; posters are
   unrepresentative. Reserved for the ADR-0010 opinion-seed/feed path (real observed
   *opinions as memories*, a separate knob).
4. **Real-microdata disposition matching + constrained LLM elaboration (chosen).**
   NN-match each census row to a real donor respondent, borrow real disposition items
   (model-free), then use the LLM only to *elaborate* those fixed inputs into an
   enactable persona — never to invent the disposition.

## Decision

A **re-runnable two-stage seeding pipeline** producing a versioned, cached persona
corpus the simulation loads:

- **Stage 1 — dataset (own engineering notebook).** ACS demographic sample + **NN
  disposition match** to a donor survey carrying both overlapping demographics and
  disposition-adjacent items (personality battery / value scale / attitude items).
  **1-NN or distance-weighted sampled match, never k-averaging** — averaging recreates
  the stereotype centroid we are eliminating. Writes a joint dataset
  (`data/persona_seed_dataset_n100.csv`, schema in the P6a brief). Multi-donor fusion
  is supported (merge several surveys via demographic grouping) with the caveat that
  each donor adds a CI assumption and cross-donor item pairs carry only
  demographically-induced correlation, not empirical.
- **Stage 2 — persona pipeline (src-first, notebook orchestrates).** Per row:
  1. **Backstory** — LLM writes a short narrative *constrained by* the fixed row +
     disposition items (enacts the real numbers; must not invent new strong traits).
  2. **Reflection → anchor (Variant A)** — LLM, as psychologist/economist, synthesizes
     temperament / risk posture / values **from the real disposition items** (not the
     backstory), producing a compact ~2-line statement. This is the *compression* step:
     as more donor items are fused in, it distills the growing bundle into a coherent
     always-on anchor.
  3. **Assemble & seed.** Reflection → the always-on `Persona.values`/`dispositions`
     anchor (R7). Backstory + demographic facts + first-person disposition statements →
     **seeded memories at t=0** (importance set at creation, recency anchored to sim
     start). Provenance (matched respondent id + match distance, generating
     model/prompt/seed) logged for R17/R29.
  Writes `data/personas_corpus_n100.json`.
- **`Population.from_corpus(...)`** loads the cached corpus — no seed-time LLM calls at
  run time, so simulations are free and deterministic. The corpus is an R17-versioned
  artifact (seed + content hash); a run's config cites which corpus it used.
- **Placement is hybrid:** reflection → always-on anchor; backstory/facts/disposition
  statements → retrievable memory. (Anchor resists R7 generic-voice collapse; memory
  keeps the opinion traceable via R29.)
- **Run design (P6b):** a 3-arm ladder over the *same* draws — **null** (generic ×100,
  the collapse floor) / **demographic-only** (the stereotype arm) / **full-pipeline**
  (real disposition). `null→demographic` isolates demographic differentiation;
  `demographic→full-pipeline` **directly tests the stereotype thesis** — does demo-only
  show collapsed within-cell variance / exaggerated between-cell separation vs. the
  real-disposition arm?
- **Phase split:** **P6a** = the pipeline (this ADR's build); **P6b** = the N=100 scale/
  infra run consuming P6a's corpus. One unknown per gate.

## Why

Real-microdata matching is the only option that restores true within-cell disposition
variance *without* making the persona layer a model-bias source — the LLM elaborates
fixed real inputs rather than inventing them. Reflection-from-items (not from the
backstory) keeps the always-on anchor grounded in real data and avoids compounding
model confabulation, while giving the anchor a form the model can actually enact and a
compression role that scales as donor surveys are fused. Caching to a versioned corpus
keeps runs deterministic and cheap while the expensive seed-time generation happens
once, deliberately, and re-runnably.

## Consequences

- **Locks in** a donor disposition survey with demographic overlap to the ACS fields
  (dependency: sourced separately, in progress) and the two-stage cached-corpus build.
- **Supersedes** the demographics-only draft of this ADR; demographic-only survives
  only as the middle *arm* of the P6b run (now an instrument to demonstrate the
  stereotype bias, not the estimator).
- **Requires** the joint-dataset schema contract (P6a brief) so the disposition agent
  has a target, and `Population.from_corpus` + a corpus artifact format.
- **Revisit if** no donor survey offers adequate demographic overlap (fall back:
  coarsen matching keys, or a smaller disposition battery), or if the elaboration steps
  prove to re-inject material bias at n=100 (tighten the constraints, or shrink the
  backstory toward raw disposition statements).
- LLM-generated dispositions and X/LinkedIn data remain out of the persona layer
  (ablation / opinion-seed path respectively).

## Rules touched

Reinterprets **R7** — anchoring is real-microdata disposition (model-free), not
model-inferred, resolving the stereotype failure mode R7 warns about. Uses **R2/R19**
(disposition seeded into private memory at t=0), **R16** (null floor + demographic-only
as the stereotype arm), **R14/R15** (voice-cluster / within-cell dispersion readout),
**R17** (persona corpus versioned + cited per run), **R29** (match provenance +
generation params logged, so an opinion traces to real data, not model invention).
Supersedes the demographics-only draft of ADR 0016.
