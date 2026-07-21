# ADR (Architecture Design Record) 0018: REBUT action — active pushback

Status: proposed
Phase: 6 (action-space maturity)

## Spike question

The simulation's vote-convergence is driven by the SPEAK contagion (agents broadcast
stances, hearing stances drives conformity). ADR 0017 added SHARE_CONSIDERATION so
agents can circulate *reasons* with no vote. This ADR asks the complementary question:
**does giving a minority an explicit way to *resist* — to push back against the
prevailing view rather than only quietly hold or flip — help disparate factions
persist (R11)?**

## Context

Through ADR 0017 the action space is SPEAK / ABSTAIN / SHARE_CONSIDERATION. A dissenting
agent's only moves are to SPEAK its own stance (an *independent* statement, not a
counter) or ABSTAIN. It has no way to *object* to what it heard. Real deliberation has
active disagreement, and R11 (committed-minority persistence) is about factions holding
against a majority — which is easier to express when an agent can rebut, not just
restate. R24 requires any new action ship with its own deterministic resolution.

## Options considered

1. **No REBUT — reuse SPEAK.** A dissenter just SPEAKs its stance. Simple, but the
   discourse carries no signal of *contestation*; listeners can't tell an endorsement
   from a rebuttal, and the model has no affordance to actively resist.
2. **REBUT with no stance (a pure objection, like a negative consideration).** Would
   make it mechanically identical to SHARE_CONSIDERATION (memory write, no tally) —
   redundant, and a rebuttal naturally advocates *for* a position.
3. **REBUT that states a stance, framed as pushback (chosen).** Carries `stance` +
   `utterance` like SPEAK, but its resolution frames the memory as active disagreement
   and it *does* tally the stance (it is still an expressed position). Distinct from
   SPEAK by framing/intent, from SHARE_CONSIDERATION by carrying + tallying a stance.

## Decision

Add **REBUT** to the closed action enum (R23); bump `ACTION_SPACE_VERSION` 2 → 3.

- **Payload:** `stance` (the position being defended) + `utterance` (the counter-argument).
  Both reuse existing schema fields; the enum auto-admits the new type. `agent.act`
  already threads `stance`/`utterance` through, so no decode change is needed.
- **Game-Master resolution (R24):** `_resolve_rebut` writes a `MemoryWrite` (kind
  `KIND_HEARD`, text `"{actor} pushed back: {utterance}"`) into each neighbor's store
  **and** emits a `WorldUpdate` for the stance — because a rebut still states a
  position, unlike a stanceless SHARE_CONSIDERATION. Malformed (missing stance or
  utterance) → no-op, like a malformed SPEAK.
- **Metrics:** a REBUT counts as an *expressed stance* — `latest_speaks` /
  `stance_distribution` / `latest_utterances` include SPEAK and REBUT (via
  `_STANCE_ACTIONS`); `action_space_adequacy` reports `n_rebut` / `rebut_rate` and folds
  rebut stances/utterances into coverage and uniqueness, and the R27 flags treat
  `n_speak + n_rebut` as the stated-position count.

## Why

Framing-plus-tally is the reading that keeps REBUT a genuinely distinct action (its own
resolution branch and discourse signal) without duplicating either SPEAK (independent
statement) or SHARE_CONSIDERATION (stanceless reason). It gives a minority the explicit
resistance move R11 is about, while remaining honestly a "vote" so the tally and
metrics stay coherent.

## Consequences

- The action space is now SPEAK / ABSTAIN / SHARE_CONSIDERATION / REBUT (v3). The
  effect on convergence is an empirical question for the loop-mode work (Part B): a
  richer, reason-and-pushback discourse only changes dynamics if the tick loop actually
  *uses* these actions (today agents still default to SPEAK). Whether REBUT should keep
  tallying under a "deliberate now, survey at the end" loop is revisited there.
- REBUT tallying means it can *reinforce* a majority too (not only a minority), so it is
  not a one-way anti-convergence lever — its effect depends on who uses it.

## Rules touched

Implements a new closed, versioned action type (R23) with an explicit deterministic
resolution (R24); serves the committed-minority / faction-persistence concern (R11);
kept schema-constrainable (R20) and visible to the R27 action-space-adequacy gate.
Supersedes nothing; extends ADR 0008 (action space v1) and ADR 0017 (SHARE_CONSIDERATION).
