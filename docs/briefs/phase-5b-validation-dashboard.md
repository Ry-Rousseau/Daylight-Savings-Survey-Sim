# Brief — Phase 5B: Validation dashboard

**Phase:** 5 (unit B of 3) · **Rules activated:** R14 (divergence metric defined before the run), R15 (logged/derived continuously per tick), R16 (null-model baseline), R17 (versioned by config) · **Gate:** R27 (action-space adequacy, checked *separately*) · **Layer:** Validation & Metrics.

## Context

5A made **persona content the controlled variable** (thick NYC cast vs the thin
`NULL_PERSONA`) and added the R8 drift probe. That unblocks the R16 null-model
baseline, which is only meaningful once persona is controlled (ADR 0012). 5B builds
the rest of the R14 layer on top of the P4 kernel `metrics.homogeneity`.

**What the P4 kernel already is:** a *categorical* stance-concentration read
(dominant-share / distinct / normalised entropy) over the discrete SPEAK stances,
plus `stance_distribution` / `homogeneity_trajectory` reading the run log offline.

**What 5B adds and why it isn't redundant.** R14 also names *pairwise opinion
distance* and *cluster count*. Categorical stance concentration is blind to
**within-stance wording homogenization**: two agents can both pick "permanent DST"
while their *language* collapses from two distinct voices into one — or, conversely,
hold the same stance in genuinely different terms. So 5B measures divergence over the
**SPEAK utterance embeddings** (BGE-small, already local): mean pairwise cosine
dispersion + a threshold-based cluster count. This is the continuous complement to
the categorical read, and it reuses the same offline-log discipline.

**R27 is a separate gate, not part of the homogeneity number.** A narrow action
space (SPEAK/ABSTAIN only, ADR 0008) can *cap observable divergence* in a way the
R16 null baseline will not catch, because the ceiling is set before the null
comparison runs. So the action-space-adequacy check is built as its **own
diagnostic** (abstain rate, stance coverage, utterance uniqueness/dispersion) and
must be read *before* any 5C convergence number is trusted — exactly as the standing
open thread requires.

**Boundary with 5C.** 5B is **source + deterministic tests only** — the measurement
instruments. The live null-vs-thick run and the plotnine dashboard are 5C. Keeping
5B network-free keeps the metrics reviewable in isolation from a live run.

## Definition of done

- `metrics.py` gains, all reading the durable log offline (R15) and network-free at
  the math core:
  - `pairwise_dispersion(vectors)` — mean pairwise cosine distance (continuous R14).
  - `cluster_count(vectors, *, threshold)` — single-linkage components (R14 cluster
    count); threshold is an explicit, documented, tunable parameter.
  - `latest_utterances(run, tick=)` + a shared `latest_speaks` (refactor
    `stance_distribution` onto it — behaviour unchanged, existing tests still pass).
  - `utterance_divergence(run, embedder, *, tick=, threshold=)` — the embedding read
    at a tick.
  - `divergence_trajectory(run, embedder, ...)` — the **continuous per-tick
    dashboard** merging categorical (dominant-share/entropy) + embedding
    (dispersion/cluster-count) signals, one row per tick (R15). Embeds each distinct
    utterance once.
  - `divergence_summary(run, embedder, ...)` — endpoint metric bundle, so a notebook
    computes it for the **null** and **thick** runs and compares (R16).
- `action_space_adequacy(run, *, stances=, embedder=)` — the **R27** diagnostic,
  reported separately from homogeneity, with obvious-degeneracy flags.
- New public surface exported from `polis`.
- Deterministic suite green (existing + new), including the unchanged
  `stance_distribution` behaviour after the refactor.

## Prerequisites

5A complete (branch `phase-5a-persona-depth`). BGE-small embeddings local. The P4
run log schema (`EVENT_ACTION` payloads carry `stance` + `utterance`).

## Ordered tasks

1. `metrics.py` — `latest_speaks` refactor + `latest_utterances`; `pairwise_dispersion`,
   `cluster_count` (numpy, unit-normalised similarity matrix). TDD.
2. `metrics.py` — `utterance_divergence`, `divergence_trajectory` (with a per-text
   embed cache), `divergence_summary`.
3. `metrics.py` — `action_space_adequacy` (R27).
4. Extend `tests/test_metrics.py` (fake embedder + `FakeRun` with utterances).
5. `__init__.py` exports; `docs/adr/0014-*.md` (metric extension + R16/R27 boundary).

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest -q` green; the P4 `stance_distribution` /
  `homogeneity_trajectory` tests pass unchanged after the refactor.
- Math proven on synthetic vectors: identical utterances → dispersion 0, cluster
  count 1; two tight groups → cluster count 2; dispersion rises with spread.
- `divergence_trajectory` yields one row/tick with both categorical and embedding
  fields; `action_space_adequacy` flags an all-abstain or single-stance run.

## Hand-off pointer

Update `status.md` (5B done, 5C next), leave `PHASE_PLAN.md` P5 open. 5C inherits
the full instrument set: run the thick cast vs `NULL_PERSONA` (R16), plot the
`divergence_trajectory` dashboard + the R8 drift trajectory, **read
`action_space_adequacy` first** (R27), and write the checkpoint answering the spike
(minimum persona content vs collapse).
