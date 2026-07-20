# Checkpoint — Phase 5: Persona depth + validation wiring

**Date:** 2026-07-20 · **Status:** ✅ complete (DoD met) · **Branch:** `phase-5a-persona-depth` (uncommitted; carries 5A+5B+5C) · **ADRs:** 0013 (persona schema + drop self-hosting), 0014 (validation dashboard)

## What it proves

Phase 5 thickened two axes at once — Persona (R7–R9) and Validation (R14–R17) — split
into three units so each is reviewable on its own: **5A** persona depth, **5B** the
validation dashboard, **5C** the live DoD run. The DoD ("full validation dashboard
running against a null-model baseline") is met, and the spike (*what minimum persona
content prevents identity collapse over N ticks?*) has a clean, quantitative answer.

**The result — persona depth protects the *voice*, not the *vote*.** A single
controlled contrast (12 agents × 5 ticks, `small_world`, `qwen/qwen3-32b`, 0 decide
failures): a **thick** arm (value/disposition-anchored NYC cast) vs a **null** arm
(thin `NULL_PERSONA`, *identical* memories/topology/seed/temperature — persona content
the only difference, the R16 baseline). Both converge to the **same stance**
(dominant-share → 1.00, permanent standard time). The categorical metric alone would
call both "converged." The embedding axis (5B) shows they are opposites:

| endpoint (tick 4) | thick | null |
|---|---:|---:|
| dominant share (categorical) | 1.00 | 1.00 |
| pairwise dispersion — voice (R14) | **0.204** | **0.121** |
| cluster count / 12 agents (R14) | **8** | **1** |
| mean identity drift from baseline (R8) | **0.127** | **0.191** |

The thin personas collapse into **one voice cluster** (5→1, dispersion 0.224→0.121) —
the R7 "generic assistant voice," quantified. The thick personas hold **8 distinct
voice-clusters among 12 agents** and drift ~33% less from their tick-0 selves. R27 was
read first: both arms speak every tick (abstain 0.00), high utterance uniqueness
(1.00 / 0.97), no degeneracy flags — the numbers are genuine convergence, not an
action-space artifact. Results: `data/phase5_persona_validation.csv`,
`data/phase5_drift.csv`; notebook `notebooks/experiments/phase5_persona_validation.ipynb`.

**R9 confirmed in the same run.** Thick personas did *not* prevent stance consensus —
the population still converged collectively despite per-agent anchors. Persona
diversity preserved *how* agents spoke, not *what* they concluded (the topology/memory
dynamics drove that, as P4 predicted). Diversity is necessary but not sufficient; the
two metric axes disagreeing is the finding.

Deterministic proof (no network): **132 tests pass** (89 P0–4 + 43 new — persona 6,
drift 12, personas_nyc 5, metrics +19, plus the P4 metric tests unchanged after the
`latest_speaks` refactor).

## What's live

- `src/polis/persona.py` — `Persona` gains `values`/`dispositions` anchors (R7);
  empty persona byte-identical to P0–4 (regression-pinned) → thin persona survives as
  the R16 null baseline.
- `src/polis/prompts.py` — `persona_system(description, values, dispositions)` composes
  the anchoring prompt; empty → the exact P0–4 string.
- `src/polis/personas_nyc.py` — hand-authored thick NYC cast (`NYC_CAST`: morning /
  evening / ambivalent / low-conviction spread + paired opinion seeds), `NULL_PERSONA`,
  `null_cast(n)`.
- `src/polis/drift.py` — R8 probe: `capture_baseline` / `probe_drift` over embeddings
  of the free-text *reason* (voice), `remember=False` so measuring doesn't perturb
  memory; retries then **skips** a flaky agent (a diagnostic tolerates a missed probe).
- `src/polis/metrics.py` — full R14 layer on the P4 kernel: `pairwise_dispersion`,
  `cluster_count` (single-linkage, explicit cosine threshold), `latest_utterances` +
  shared `latest_speaks`, `divergence_trajectory` (R15 per-tick, both axes),
  `divergence_summary` (R16 null-vs-thick bundle), `action_space_adequacy` (R27 gate).
- `src/polis/agent.py` — `answer(..., remember=True)` flag (the non-mutating probe
  path).
- `src/polis/simulation.py` — persona `values`/`dispositions` versioned in the run
  config (R17).
- Tests: `tests/test_persona.py`, `tests/test_drift.py`, `tests/test_personas_nyc.py`,
  extended `tests/test_metrics.py`.
- Notebook executed live (0 decide failures), plots + tables + verdict in place.

## DoD status

| DoD clause | State |
|---|---|
| Value/disposition-anchored personas (R7) | ✅ `Persona` anchors + thick NYC cast |
| Persona strength measured, not assumed (R8) | ✅ drift probe; thick drift 0.127 < null 0.191 |
| Full R14–17 metrics logged per tick | ✅ `divergence_trajectory` both axes, per tick |
| Running against a null-model baseline (R16) | ✅ thin `NULL_PERSONA` arm, memory held |
| R27 adequacy checked separately, first | ✅ §1 gate, no flags, read before §2–4 |
| deterministic suite green | ✅ 132 passed, 0 failures |

## What P6/P7 need

- **P6 (scale to 100):** the run was N=12. Re-run the thick-vs-null contrast at N=100
  and confirm the voice-diversity gap (8 vs 1 clusters) holds — cluster count may need
  a swept threshold at larger N. ACS PUMS census→persona wiring (deferred from 5A) is
  the P6 seed pipeline for 100 demographically-realistic agents; R7 anchors layer on
  top of the census demographics.
- **P7 (survey subsystem):** the Layer-5 `run_survey` fan-out is **not resilient** to
  the intermittent Qwen3 whitespace-blob quirk — one bad agent raises and aborts the
  survey (the thick §5 survey was skipped by the notebook guard). Harden `run_survey`
  with per-agent retry/skip when the survey subsystem is matured. Also: a per-tick
  drift *trajectory* needs a `Simulation` step API (drift is currently baseline→endpoint).

## Debt / notes

- **No LLM self-hosting, ever (ADR 0013, amends 0002).** OpenRouter `qwen/qwen3-32b`
  for the whole project; the P5+ vLLM `Qwen3-32B-AWQ` leg is dropped.
- **Conviction-slider seam left open, not built.** The end-goal slider (seed-time
  X-corpus injection → generic empty, sibling of the ADR-0010 runtime feed) rides on
  the new persona anchors + opinion seeds; it is a documented seam, not a P5 deliverable.
- **Qwen3 whitespace quirk is intermittent, not systematic** — 0 decide failures at
  `max_retries=4`; only the un-retried survey path is exposed. Bump retries or harden
  `run_survey` if it bites at N=100.
- **Cluster count is threshold-dependent** (0.15, logged per R17); the 8-vs-1 gap is
  robust to the exact value but absolute counts are not portable across thresholds.
