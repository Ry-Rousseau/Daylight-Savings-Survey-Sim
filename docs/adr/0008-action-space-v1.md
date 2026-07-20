# ADR 0008: Action space v1 — SPEAK / ABSTAIN + Game-Master resolution

Status: accepted
Phase: 2

## Spike question

What is the minimal action schema and resolution logic for one interaction type
(R23/R24), and how does the always-valid no-op (R25) fit?

## Context

Phase 2 thickens exactly one axis — interaction — per the walking-skeleton
principle: isolate one new unknown. The unknown is whether the Game-Master /
world-state / provenance-logging machinery works end to end for a *single*
resolved interaction, not whether a rich dialogue system works. R23 requires the
action space to be a closed, versioned enumeration with structured payloads (so it
stays schema-constrainable, R20, and quantitatively loggable, R14–R17); R24
requires every shipped action type to have a deterministic resolution function
before it ships; R25 requires a no-op to always be available.

## Options considered

1. **SPEAK + ABSTAIN, one-directional** — one content-exchange action (express a
   DST stance with a short utterance) plus the no-op. The listener receives the
   utterance as a `heard` memory; no same-tick reply. Minimal; isolates the
   machinery as the single new unknown.
2. **Add a listener reaction/reply** — richer dialogue, but introduces a second
   unknown (turn-taking / reply resolution) into a phase meant to isolate one.
3. **Free-text actions** — maximally expressive, but violates R23 (not a closed
   enumeration), breaks constrained decoding (R20) and quantitative logging, and
   would let action-space adequacy (R27) drift invisibly.

## Decision

**Option 1.** `src/polis/actions.py` defines a closed `ActionType` enum
(`SPEAK`, `ABSTAIN`) with `ACTION_SPACE_VERSION = 1` and a json_schema for
constrained decoding (`action_type` required; `stance` constrained to the DST
option set; `utterance` free string — stance/utterance validated by the Game
Master, since OpenRouter soft-enforces). `src/polis/game_master.py` resolves
deterministically (R24): SPEAK → one `MemoryWrite` (kind=`heard`) into each
neighbour's store + one `WorldUpdate` (stance-tally increment); a malformed SPEAK
degrades to a no-op; ABSTAIN → no effects (R25). Cross-agent memory originates
*only* from a GM-resolved effect (R2).

## Why

A two-action space is the smallest thing that exercises the full path
(decide → resolve → apply → log with provenance) while keeping the phase's single
unknown single. Versioning the enum (`ACTION_SPACE_VERSION`) makes a later
expansion a visible, logged schema change rather than a silent widening. Keeping
resolution deterministic and non-LLM (R24) means the interaction layer is fully
testable without a network and keeps the LLM's role to *proposing* structured
actions, not adjudicating their consequences.

## Consequences

- **R27 (action-space adequacy)** is deliberately narrow now and must be tracked
  *separately* from any homogeneity metric: a 2-action space can suppress
  observable divergence in a way the R16 null-model baseline won't catch, because
  the ceiling is set before that comparison runs. Flagged for P4/P5 — widen the
  action space *and* re-check adequacy before reading convergence results.
- **R26 (topology-mutating actions)** — tie formation/dissolution — are **not**
  in v1; they arrive with pluggable topology at P4. The run log's `event_type`
  set is shaped so those land as their own stream (a distinct event type), never
  folded into `action`, per R26.
- The speaker does not currently write its *own* memory of having spoken; if
  cross-tick speaker coherence needs it, add a self-writeback (R19-style) in a
  later phase — noted, not built, to keep P2 tight.
- `LLMClient.choose()` is unchanged; `decide()` is a sibling sharing one private
  json_schema helper (contract of `choose` preserved, guarded by the live test).

## Rules touched

R23 (closed versioned action space), R24 (deterministic GM resolution), R25
(abstain no-op), R20 (constrained decoding, soft on OpenRouter), R2 (cross-agent
memory only via GM effect). Constrains R26/R27 for later phases.
