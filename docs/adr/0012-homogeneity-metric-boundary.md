# ADR 0012: The P4/P5 homogeneity-metric boundary — a minimal tested instrument now

Status: accepted
Phase: 4

## Spike question

Phase 4's DoD requires a *"measurably different homogeneity metric"* across
topologies — but the metric layer (R14–R17) is filed under Phase 5. What is the
minimum, credible measurement instrument that lets P4 answer its own spike without
either front-running P5 or inventing an ad-hoc number the way R14 forbids?

## Context

The Phase-4 spike is *"how much does the divergence metric change under different
graph structures?"* — so the homogeneity metric is the phase's **instrument**, not
an incidental detail. The DoD insists on *"objectively checkable, not vibes-based."*
Yet `PHASE_PLAN.md` puts the full metric layer at P5, and `Run.metrics` is a
documented stub. This is a real seam in the plan, and it cuts two ways:

- **Under-building** — computing a number inline in the notebook after the run —
  violates R14's intent (*"defined before any full run, not after"*), which exists
  specifically to prevent post-hoc, un-versioned, un-reusable metrics. P4's headline
  claim would become the vibes-based thing the DoD prohibits.
- **Over-building** — pulling all of R14–R17 forward — blurs the boundary the plan
  drew on purpose. In particular the R16 null-model baseline ("no persona /
  non-interacting agents") only *means* something once personas are the controlled
  variable, and personas stay thin until P5 deepens them (R7–R9). Running R16 now
  baselines against a persona layer about to change underneath it.

## Options considered

1. **A small, tested, defined-before-run metric module now; full layer at P5
   (chosen).** `src/polis/metrics.py` with `homogeneity(distribution)` (dominant-
   share, distinct-count, normalised entropy), fed by `stance_distribution(run,
   tick)` read from the logged SPEAK stances. P5 extends this same kernel.
2. **Ad-hoc metric inline in the DoD notebook.** Fastest, but un-tested,
   un-versioned, non-reusable, and exactly what R14 forbids. Rejected.
3. **Full R14–R17 dashboard + R16 null baseline now.** Honours R14 maximally but
   front-runs P5, and the R16 baseline is premature against thin personas. Rejected:
   measures against a moving target and doubles the phase.

### Sub-decision: headline signal = logged-stance trajectory

Two population signals were available: (a) the SPEAK stances already in the run log
(free, per-tick, self-selected — an ABSTAIN contributes no stance), and (b) a
`population.survey()` distribution (the deliverable's own op, but an endpoint
snapshot costing an extra fan-out per config). Chosen: **(a) as the headline** —
per-tick means it shows *when* concentration happens (the R15 rationale) and is
cheap enough to run many configs — **with (b) as a single endpoint confirmation**
per config.

## Decision

**Option 1**, headline = logged-stance trajectory.

`homogeneity(distribution, *, support=None)` returns `dominant_share` (plurality
fraction; the headline), `distinct` (stances present), and normalised Shannon
`entropy` (0 = consensus, 1 = uniform; normalised by `log(support)` when the option-
set size is given so consensus onto one of K options reads as more homogeneous than
an even split across K). `stance_distribution(run, tick=None)` tallies each agent's
most-recent-SPEAK stance as of a tick; `homogeneity_trajectory(run)` is the per-tick
curve. All pure and unit-tested (`tests/test_metrics.py`).

The module docstring states the P4/P5 boundary explicitly: embedding pairwise-
distance and cluster count (the rest of R14), the continuous dashboard, and the R16
null-model baseline are P5 and build on `homogeneity`.

## Why

P4 cannot honestly claim "measurably different" without a defined, tested measure,
and R14 forbids defining it after the run — so a *real* metric function, written and
tested before the DoD run, is the minimum that satisfies both the DoD and R14's
intent. Keeping it to stance-concentration (not embedding geometry, not a null
baseline) is what keeps it from becoming the P5 phase early: it is the shared kernel
P5 extends, so nothing is thrown away. The logged-stance trajectory is the better
instrument for *this* spike than a survey endpoint — the spike is about *how much*
and *when* structure changes convergence, which is a trajectory question.

If a later result looks suspicious, this ADR is where to check what the P4 number
did and did not claim: it is stance concentration over self-reported SPEAK stances,
not internal-belief divergence, and not baselined against a null model.

## Consequences

- **`metrics.homogeneity` is the shared kernel.** P5 adds embedding-distance /
  cluster-count metrics, the R16 null baseline, and the continuous dashboard on top
  of it; `Run.metrics` stays a stub until then.
- **The P4 finding is bounded by action-space adequacy (R27, ADR 0008).** A narrow
  SPEAK/ABSTAIN space can suppress observable divergence; a low homogeneity reading
  is *stance concentration*, not proof of internal consensus. This check is separate
  and lands at P5.
- **ABSTAIN self-selection.** An agent that only ever abstains contributes no
  stance; the denominator is agents with an expressed position. Acceptable for a
  concentration read; noted so P5's richer metric can handle abstention explicitly.
- **No new run-log surface.** The metric reads the existing `action` event stream
  offline; nothing new is logged for it this phase.

## Rules touched

R14 (homogeneity metric defined before the run — minimally, this phase). Defers the
rest of R14 (embedding distance, cluster count), R16 (null-model baseline), and the
R15 continuous dashboard to P5. Bounded by R27 (action-space adequacy, ADR 0008).
Consistent with R17 (the run whose log the metric reads is itself versioned).
