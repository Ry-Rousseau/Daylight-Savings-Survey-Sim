# ADR 0007: Within-tick update scheme — simultaneous by default

Status: accepted
Phase: 2

## Spike question

Within a tick, when does each agent read vs write state — simultaneously (all
decide from the same pre-tick snapshot) or sequentially (each sees the prior
agents' effects same-tick)? And how is that choice recorded (R28)?

## Context

The custom tick loop (R22) schedules agents, each of whom reads memory, acts, and
— via a resolved SPEAK — writes a `heard` memory into another agent's store. The
*interleaving* of those reads and writes changes how much within-tick contagion
occurs: if agent 1's utterance lands in agent 2's memory before agent 2 decides,
agent 2 can be swayed within the same tick. R28 requires this be an explicit,
configurable run parameter, recorded, never an implementation accident, precisely
because it changes convergence — a homogenization-relevant knob, not a detail.

## Options considered

1. **Simultaneous (snapshot decide → atomic apply)** — every agent decides from
   the same pre-tick state; all Game-Master effects are applied after the decide
   phase. No agent hears another within the tick. Lower within-tick contagion; a
   cleaner baseline for isolating *cross-tick* dynamics. Slightly less "realistic"
   as a conversation.
2. **Sequential (interleaved)** — agent decides, effects apply immediately, next
   agent decides seeing them. Higher within-tick contagion; more like a live
   conversation, but muddies the convergence signal (within- and cross-tick
   contagion are entangled) and makes the result order-dependent.

## Decision

**Simultaneous is the default**; sequential is selectable via
`DynamicsConfig.update_scheme ∈ {"simultaneous", "sequential"}`. The chosen scheme
is written into the run config (R17) and hashed, so any run's trajectory is
attributable to it. Implemented in `src/polis/simulation.py` as two explicit
tick paths (`_tick_simultaneous` / `_tick_sequential`) dispatched on the config.

## Why

The project treats convergence as a first-class, measured phenomenon; the default
should minimize *incidental* contagion so that observed homogenization is
attributable to deliberate dynamics (topology, exchange volume — P4) rather than
to a within-tick ordering artifact. Simultaneous update gives that clean baseline
and removes scheduling-order sensitivity from the default. Sequential remains one
config flag away for when conversational realism is the object of study. Making
the knob explicit and logged is the R28 requirement itself — the decision here is
which *default*, and it's the conservative one.

## Consequences

- Default runs are order-insensitive within a tick; comparisons across runs don't
  silently depend on roster order.
- The two schemes are behaviorally distinguishable and tested
  (`tests/test_simulation.py`): under sequential, agent 2's retrieval provenance
  includes agent 1's same-tick utterance; under simultaneous it does not.
- No new ARCHITECTURE rule is needed — R28 already owns this; the design note's
  "add a tick-ordering rule" is satisfied by R28 as written.
- Revisit the *default* if a later phase makes conversational realism the primary
  research target rather than a clean convergence baseline.

## Rules touched

R28 (explicit configurable update scheme, recorded), R17 (scheme is part of the
hashed run config), R22 (this is the custom tick loop half).
