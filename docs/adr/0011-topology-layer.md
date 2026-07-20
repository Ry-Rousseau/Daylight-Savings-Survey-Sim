# ADR 0011: Phase 4 topology — pluggable seeded graphs, static-swappable not agent-mutated

Status: accepted
Phase: 4

## Spike question

How much does the divergence metric change under different graph structures at
this N (R4, R10–R13)?

## Context

Since Phase 2 the interaction topology has been a single seam: `Simulation.topology`
is a `Callable[(agent_id, roster) -> list[str]]` defaulting to `fully_connected`
(everyone hears everyone). That is the R10 anti-pattern made concrete — full
exposure biases toward global consensus — and it is the only graph the engine can
express. Phase 4 must make topology a *pluggable, swappable, seeded* parameter
(R4/R13) and provide sparse/structured alternatives (R10), then show that swapping
it changes the homogeneity outcome on identical personas.

Constraints in play:
- R2 forbids cross-agent shared mutable state; the graph is read-only global
  structure, queried per speaker, so it sits cleanly on the environment side.
- R28/ADR 0006: only the resolve/apply phase (serial, single-writer) touches the
  graph; the concurrent decide phase (P3) must not, or the single-writer invariant
  breaks. Topology is queried in `_resolve`, which is serial — safe.
- R17: whatever graph ran must be reconstructable from the logged config.

## Options considered

1. **A `Topology` class hierarchy, drop-in for the existing callable seam
   (chosen).** Each graph is an object that is *callable* `(agent_id, roster)`,
   builds its adjacency once (frozen, cached) and deterministically from a `seed`,
   and exposes `to_config()` for R17. `FullyConnected`, `RingLattice`, `SmallWorld`
   (Watts–Strogatz), `StochasticBlock`. Keeps the free `fully_connected` function as
   the default (back-compat). Swap = assign a different object to `sim.topology`.
2. **Precomputed adjacency dicts passed as plain callables (closures).** Simpler
   types, but no structured `to_config()` (R17 would log only `"custom"`), and
   "freezable/swappable" (R13) becomes an informal convention rather than a property
   of a named object. Rejected: loses the versioning and the identity the
   counterfactual needs.
3. **A graph library (networkx) under the hood.** More graph generators for free,
   but a heavyweight dependency for four small generators, and its objects would
   still need wrapping to fit the callable seam and `to_config()`. Rejected: not
   worth the dependency at this N; revisit if P5+ needs exotic generators.

### Sub-decision: static swappable topology, **not** agent-mutated ties (R26)

The DoD's counterfactual is *"same personas, different graph"* — a graph that is
fixed within a run and swapped *between* runs (R13). R26 (topology-*mutating*
actions — tie formation/dissolution as agent moves) is a different, heavier
capability: it expands the action space (R23/R24 resolution logic), changes the
graph mid-run, and needs its own logged event stream. Building it now is not
required to answer the spike and would couple two unknowns.

## Decision

**Option 1**, plus the sub-decision to build **static** topology only.

`src/polis/topology.py` adds `Topology` (base: callable + cached-frozen symmetric
adjacency + `to_config`) and four concrete graphs, each deterministic given `seed`
and undirected (an edge is a mutual channel, matching SPEAK → MemoryWrite-per-
listener semantics):

- `FullyConnected` — the R10 high-consensus-pressure control.
- `RingLattice(k)` — k-regular ring: sparse, highly clustered, long paths.
- `SmallWorld(k, p)` — Watts–Strogatz: ring + edge rewiring; sparse, short paths.
- `StochasticBlock(n_blocks, p_in, p_out)` — clustered communities (R10 "preserve
  disparate communities"), with `block_assignments()` exposed for seeding a
  committed faction within one block.

Two dynamics knobs ship alongside, since the phase plan activates R11/R12 here:

- **Exchange volume (R12):** `DynamicsConfig.exchange_volume: int | None` caps how
  many of a speaker's neighbours actually hear a given SPEAK, subsampled
  deterministically from `(seed, tick, agent_id)`. Consensus pressure becomes
  tunable independently of graph density. `None` = full reach (prior behaviour).
- **Committed minority (R11):** `Agent.committed_stance` makes an agent SPEAK a
  fixed stance deterministically with **no model call and empty provenance** — an
  immovable faction. The committed roster is versioned in the run config.

R17: `config["topology"]` is now the graph's structured `to_config()` (name +
params + seed); `exchange_volume` and the committed roster are versioned too.

R26 seam: `EVENT_TIE_CHANGE` is reserved in the run-log vocabulary so tie-mutating
actions get their own stream when built — but no such action ships this phase.

## Why

The value the phase must deliver is a topology that is genuinely pluggable (any of
four graphs at runtime), seeded (reproducible + versioned), and swappable for the
same-personas/different-graph counterfactual — and the demonstration that this
changes homogeneity. A class that *is* the callable seam gets this with zero change
to the tick loop (the `Topology` type alias already accepts it) while adding the
`to_config()` identity R17 needs. Determinism-by-seed is what makes a run
reproducible and a config hash meaningful.

Holding R26 (agent-mutated ties) out keeps the phase to one unknown (does graph
*structure* change convergence) rather than two (structure + endogenous graph
dynamics). The static graph fully satisfies R13's counterfactual; endogenous ties
are a later dynamics feature with their own resolution logic and event stream.

If, mid-debug at P5, a convergence result looks off, this decision is a plausible
root cause to check: the graph is frozen per run and its exact structure is in
`config["topology"]` (name+params+seed) — rebuild it and inspect the adjacency.

## Consequences

- **Topology is now a first-class versioned parameter.** Any run's graph is
  reconstructable from its config; the config hash changes when the graph does.
- **The concurrent decide phase is untouched.** Topology is read only in the serial
  resolve phase, so ADR 0006's single-writer invariant holds. Revisit if a later
  phase parallelizes resolve/apply.
- **R26 tie-mutation is deferred**, event-stream seam reserved. Revisit when
  modelling endogenous community formation (faction dynamics beyond a fixed
  committed set).
- **Committed agents skip the model**, so a committed-heavy run costs fewer tokens
  and is partly deterministic — useful for cheap large-N structure tests, but their
  stance is by construction immovable (they hear but never update).
- **`SmallWorld`/`StochasticBlock` clamp their degree for small rosters**; at the
  DoD's N this is inert, noted for tiny-N tests.

## Rules touched

R4 (topology pluggable/swappable), R10 (sparse/clustered defaults available),
R11 (committed-minority affordance), R12 (exchange-volume tunable), R13
(freezable/swappable for counterfactuals). Constrains: R17 (graph + knobs
versioned), R2/R28/ADR 0006 (topology read only in the serial phase). Reserves the
R26 seam (`EVENT_TIE_CHANGE`) without implementing it. Interprets the metric via
ADR 0012.
