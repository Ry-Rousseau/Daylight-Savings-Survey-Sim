# ADR (Architecture Design Record) 0014: Validation dashboard — embedding divergence + R16/R27 boundary

Status: accepted
Phase: 5 (unit B)

## Spike question

Phase 5's DoD is *"full validation dashboard running against a null-model
baseline."* Unit 5B resolves: **what does the full R14 metric layer measure beyond
the P4 categorical kernel, and how are the R16 null baseline and the R27
action-space-adequacy gate wired without conflating them with the homogeneity
number?**

## Context

ADR 0012 built the P4 kernel `metrics.homogeneity` as a *categorical*
stance-concentration read (dominant-share / distinct / normalised entropy over the
discrete SPEAK stances) and deferred the rest of R14 to P5. R14 also names *pairwise
opinion distance* and *cluster count*, and the DoD names a *null-model baseline*
(R16) and — via the standing open thread — the R27 action-space-adequacy gate.

Constraints in play:
- The categorical read is blind to **within-stance wording homogenization**: agents
  can converge in *language* while their discrete stance tally is unchanged, or hold
  the same stance in genuinely different terms. A convergence study that only counts
  stances would miss the collapse R7/R9 warn about.
- R16's null baseline is only interpretable once persona is the controlled variable
  (ADR 0012) — 5A supplied that (`NULL_PERSONA` / `null_cast`).
- R27 is explicitly a *separate* check: a narrow action space caps observable
  divergence before the null comparison runs, so folding adequacy into the
  homogeneity number would hide the ceiling.
- BGE-small embeddings already run locally (used by memory retrieval + the 5A drift
  probe), so a continuous metric costs no new dependency.

## Options considered

**Continuous divergence metric:**
1. Categorical only (status quo) — misses within-stance homogenization.
2. **Embedding pairwise dispersion + threshold cluster count over SPEAK
   utterances**, reported *alongside* the categorical read. Reuses the offline-log
   discipline and the local embedder; captures voice collapse.
3. LLM-judged semantic distance — a second model call per pair, non-deterministic,
   un-versionable, and circular (using the population's model to judge its own
   divergence).

**Clustering:** k-means / HDBSCAN (sklearn dependency, non-deterministic seeding or
extra params) vs **single-linkage connected components at an explicit cosine-distance
threshold** (deterministic, dependency-free, one transparent knob).

**R16 / R27 packaging:** one combined "health score" vs **distinct instruments** —
`divergence_summary` (the comparable bundle for null vs thick) and
`action_space_adequacy` (a separate diagnostic with degeneracy flags).

## Decision

- Add to `metrics.py`, all reading the durable log offline (R15), math network-free:
  - `pairwise_dispersion(vectors)` (mean pairwise cosine distance) and
    `cluster_count(vectors, *, threshold)` (single-linkage components) — the
    continuous R14 pair.
  - `latest_speaks` (shared reader; `stance_distribution` refactored onto it,
    behaviour unchanged) + `latest_utterances`.
  - `utterance_divergence` (embedding read at a tick), `divergence_trajectory` (the
    R15 continuous dashboard merging categorical + embedding per tick, embedding each
    distinct utterance once), `divergence_summary` (endpoint bundle for the R16
    null-vs-thick comparison).
  - `action_space_adequacy(run, *, stances=, embedder=)` — the **R27** diagnostic
    (abstain rate, stance coverage, utterance uniqueness / dispersion) with
    `flags` for `all_abstain` / `single_stance` / `low_utterance_variety`.
- Cluster threshold is an explicit parameter (`DEFAULT_CLUSTER_THRESHOLD = 0.15`,
  swept in 5C, recorded per call for R17), never a hidden constant.
- 5B is source + deterministic tests only; the live null-vs-thick run and the
  plotnine dashboard are 5C.

## Why

Reporting the embedding read *beside* the categorical one is the only option that
can see within-stance collapse while keeping the well-understood dominant-share
signal — and it adds no dependency or nondeterminism. Connected-components clustering
keeps the metric reproducible with a single transparent knob, which matters because
cluster count is inherently threshold-sensitive and R17 requires the reading be
traceable to how it was computed. Keeping R16 and R27 as distinct instruments
preserves the ARCHITECTURE.md guarantee that the action-space ceiling is checked
*before* the null comparison, so a low homogeneity reading can be correctly
attributed to consensus vs a suppressed action space.

## Consequences

- `stance_distribution` now delegates to `latest_speaks`; the P4 metric tests pass
  unchanged (regression-guarded).
- Cluster count comparisons are only meaningful at a fixed threshold — the 5C
  notebook must sweep it and log the value with each run (R17).
- The embedding metric measures *utterance* voice; an all-committed or all-abstain
  run yields few/no utterances, which is exactly what `action_space_adequacy` flags
  — read it first (R27).
- `metrics` stays a submodule import (consistent with P4), not re-exported from the
  package root.
- 5C inherits the full instrument set: run thick cast vs `NULL_PERSONA` (R16), plot
  `divergence_trajectory` + the 5A drift trajectory, read `action_space_adequacy`
  first, and answer the spike (minimum persona content vs collapse).

## Rules touched

Implements the rest of R14 (pairwise distance + cluster count), R15 (continuous
per-tick dashboard), R16 (null-baseline comparison instrument), R17 (explicit,
recorded metric parameters), and R27 (action-space adequacy as a separate gate).
Builds on ADR 0012's kernel; does not re-litigate the categorical read.
