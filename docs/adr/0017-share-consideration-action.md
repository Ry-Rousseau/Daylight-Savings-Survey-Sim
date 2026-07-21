# ADR 0017: SHARE_CONSIDERATION — circulate reasons, not votes

Status: proposed
Phase: 6

## Spike question

Does circulating *reasons* rather than *votes* reduce vote-convergence? The
simulation collapses to a single stance because the only substantive action is
`SPEAK(stance, utterance)`: every tick an agent broadcasts a vote, and hearing
votes drives conformity. Can a non-stance content action attack that conformity
mechanism directly — and is it a genuine widening of the action space (R27) or
just re-skinned SPEAK?

## Context

R27 warns that a narrow action space caps observable divergence in a way the R16
null baseline won't catch, because the ceiling is set before the null comparison
runs. The v1 space (ADR 0008) is exactly two substantive moves — SPEAK and the
ABSTAIN no-op — so the only thing agents can circulate is a stance-tagged vote.
Every heard memory is therefore a vote, and the tally plus the heard stream both
push the same direction: toward the plurality. We want a lever that lets an agent
put a persona-specific *stake* into the conversation ("as a night-shift worker,
my evenings are when I'm alive") without also casting a vote for an option.

Constraints in play: R23 (closed, versioned, structured action space — no free
text), R24 (every shipped action type needs a deterministic Game-Master
resolution before it ships), R25 (abstain/no-op always valid), R20 (schema-
constrained decoding).

## Options considered

1. **SHARE_CONSIDERATION — a stanceless reason that reaches neighbours as a heard
   memory, with no WorldUpdate.** Widens the action space with one new type;
   circulates reasons over the same topology SPEAK uses, but contributes nothing
   to the stance tally. Minimal, isolates the "reasons vs votes" unknown.
2. **REBUT — a targeted reply that argues against a heard stance.** Richer, but
   introduces a second unknown (target selection + reply resolution) into a change
   meant to isolate one, and its resolution logic is not obvious. Left as a future
   action (see Consequences).
3. **Let SPEAK carry an optional empty stance.** Rejected: overloads one action
   type with two resolutions, muddies the schema (R23) and the stance read, and
   makes "was this a vote?" ambiguous in the log.

## Decision

**Option 1.** Add `SHARE_CONSIDERATION = "share_consideration"` to the closed
`ActionType` enum and bump `ACTION_SPACE_VERSION` from 1 to 2 (a visible, logged
schema change, per R23). The payload is a single first-person `consideration`
string — no `stance`. `Action.consider(...)` builds it and
`Action.is_valid_consideration()` guards it; `action_json_schema` gains an
optional `consideration` field (GM-validated, like stance/utterance).

Game-Master resolution (R24), `_resolve_consideration`: deliver one `MemoryWrite`
(kind `heard`) into **each neighbour's** store — text `"{actor_label} shared:
{consideration}"` — and **emit no `WorldUpdate`**. A malformed/empty
consideration degrades to a no-op (`[]`), exactly like a malformed SPEAK. So a
consideration reaches listeners over the topology exactly as a SPEAK does, but it
**never touches `WorldState.stance_tally`**.

## Why

The no-stance-tally decision is the whole point, not an implementation detail. A
consideration is a reason, not a vote; if it incremented the tally it would be a
SPEAK with extra words, and the R27 concern — that everything circulating is a
vote — would be untouched. By separating "put a stake into the room" from "cast a
vote," an agent can hear another's lived reason without that reason registering as
another data point on the plurality it is being asked to conform to. The
categorical stance read (`stance_distribution` / `latest_speaks` /
`latest_utterances`) counts only SPEAK stances, so a consideration is correctly
invisible to the vote metric while still shaping the memory an agent retrieves at
its next decision. This is the mechanism we want to test against convergence.

Keeping resolution deterministic and non-LLM (R24) means the new type is fully
testable without a network; versioning the enum makes the widening auditable; and
`action_space_adequacy` now reports `n_consider` / `consider_rate` so a run that
actually exercises the new action is visible in the R27 gate rather than hidden.

## Consequences

- **R27 gate extended.** `metrics.action_space_adequacy` gains `n_consider` and
  `consider_rate`, and its `all_abstain` flag no longer fires when a run is made
  of considerations (a stanceless-but-active space is not a collapsed one). Read
  consider-rate alongside abstain-rate before trusting any homogeneity number.
- **The stance tally can now under-count expressed engagement.** By design: a
  consideration-heavy tick moves memory without moving the tally, so a run's
  `stance_tally` is no longer a proxy for "how much was said." Downstream reads
  that assumed tally == activity must use the action stream instead.
- **REBUT remains a future action.** Targeted, stance-arguing replies are the
  natural next widening but carry their own unknown (target selection + reply
  resolution) and are deliberately out of scope here — noted, not built.
- **`ACTION_SPACE_VERSION = 2`** is logged in every run config (R17); a run's
  divergence is now traceable to which action-space version produced it.
- Revisit if a consideration should also carry a soft lean, or if considerations
  should decay differently from heard votes in memory (both currently share
  `HEARD_IMPORTANCE`).

## Rules touched

R23 (closed, versioned, structured action space — enum + version bump + typed
payload), R24 (explicit deterministic GM resolution, `_resolve_consideration`,
before ship), R25 (abstain/no-op still always valid; malformed consideration
degrades to it), R27 (action-space adequacy extended to measure consideration
usage). Also R20 (schema-constrained decoding gains the `consideration` field),
R2 (cross-agent memory still originates only from a GM-resolved effect), R3
(stance tally deliberately *not* touched).
