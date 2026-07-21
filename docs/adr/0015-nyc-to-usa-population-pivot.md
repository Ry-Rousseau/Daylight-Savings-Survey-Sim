# ADR (Architecture Design Record) 0015: Pivot the modelled population NYC → USA

Status: proposed
Phase: 6

## Spike question

Not the P6 convergence spike itself, but a project-level precondition that P6 forces
into the open: **which population are we actually a silicon sample *of*, given that
Phase 7 owes a calibration against real survey ground truth?** Through P0–P5 the
answer was "NYC". This ADR changes it to "the USA".

## Context

The project's headline claim is a *surveyable* silicon sample whose opinion estimate
can be checked against reality (the validation axis, Layer 4; calibration is the R21
hold-out matured at P7). That check needs a **published ground-truth DST opinion
figure** for the modelled population.

- **NYC has no such figure.** DST polling is national (Gallup, AP-NORC, Monmouth,
  YouGov all field *US* samples). There is no credible NYC-specific permanent-DST /
  permanent-standard / keep-switching breakdown to calibrate against. Under NYC the
  validation target stayed permanently "TBD / proxy" (`status.md` open threads).
- **P6 is where the population is actually *sampled*.** Until now the cast was
  hand-authored (P0–P5, 3–12 personas), so geography was cosmetic prose. P6 seeds
  ~100 personas from census microdata (delivered as a DataFrame by a separate
  pipeline), so the geographic frame stops being cosmetic and becomes the sampling
  frame — the moment to pick it deliberately.
- **The switch is cheap now, expensive later.** The persona wiring (`from_census`,
  ADR 0016) is being built this phase; making it geography-neutral costs nothing
  today. Retrofitting a US frame after P6 hard-codes NYC everywhere would be a
  rewrite.

The NYC-specific asset in the repo is `data/nyc_puma_crosswalk_2020.csv` (PUMA →
borough/neighbourhood). Under a US frame it is retired to geographic enrichment, not
load-bearing; the census DataFrame carries whatever geography field the US frame
uses (state / region / PUMA).

## Options considered

1. **Stay NYC.** Keeps every existing prose reference and the PUMA crosswalk
   load-bearing. But the validation target stays unsourceable — the project can
   scale (P6) but can never *calibrate* (P7) against a real number, gutting the
   Layer-4 claim. The homogeneity/convergence result survives; the "surveyable
   estimate of a real opinion" claim does not.
2. **Stay NYC, calibrate against a US figure as a proxy.** Dishonest: a US poll is
   not NYC ground truth, and NYC's opinion plausibly differs (denser, more transit,
   different latitude/sunset behaviour). Would bake an un-auditable frame mismatch
   into the one number the project is judged on.
3. **Pivot to the USA (chosen).** Loses NYC colour and the PUMA crosswalk's central
   role; gains a real, national, well-documented calibration target and a census
   frame (US ACS) that is standard and abundant. The convergence machinery (Layers
   1–4) is geography-agnostic and unaffected.

## Decision

Model **the USA**, not NYC, from Phase 6 onward. The persona-seeding wiring is built
**geography-neutral** (ADR 0016): it reads whatever geography column the census
DataFrame supplies and composes persona text from it generically, with no
NYC/borough hard-coding. Validation ground truth is a **published US DST poll**,
sourced and wired at P7 (deferred this phase — see the P6 brief).

Enacting doc changes (top-level framing in `CLAUDE.md`, `PHASE_PLAN.md`, the project
brief) are held until this ADR is accepted, so the pivot lands as one reviewed change
rather than drifting across files.

## Why

The deciding factor is **falsifiability of the headline claim**. A silicon sample
whose estimate can never be checked against a real figure is not the deliverable the
project set out to build; NYC forecloses that check, the USA restores it. The cost —
retiring NYC prose and the PUMA crosswalk's primacy — is cosmetic against a
convergence engine that never depended on geography. Doing it at P6 (the sampling
moment) rather than P7 (the calibration moment) avoids a hard-coded-NYC rewrite.

## Consequences

- **Locks in** a US census frame for `from_census` (ADR 0016) and a US poll as the
  P7 calibration target. The hand-authored NYC casts (`personas_nyc.py`, P5) become
  *NYC-flavoured legacy fixtures* — still valid for the P5 result already recorded,
  not the P6 population.
- **Retires** `data/nyc_puma_crosswalk_2020.csv` from a load-bearing role to optional
  enrichment; the US geography field comes from the census DataFrame.
- **Forecloses** a clean NYC-specific claim without re-sourcing NYC ground truth —
  acceptable, since none existed.
- **Revisit if** a credible NYC DST figure surfaces *and* an NYC-specific result is
  wanted, or if the census pipeline can only deliver an NYC frame after all (then the
  geography-neutral wiring still runs, but the validation target reverts to TBD).
- Requires the held doc updates (`CLAUDE.md`, `PHASE_PLAN.md`, project brief, README)
  on acceptance; `status.md` records the pivot immediately as an open decision.

## Rules touched

No Layer-1–3 rule changes — the engine is geography-agnostic. Bears on **R17**
(geography becomes a versioned run-config field via the persona set) and on the
Layer-4 validation contract (the calibration target the P7 R21 hold-out will use).
Amends the project framing in `CLAUDE.md` (population = USA, not NYC).
