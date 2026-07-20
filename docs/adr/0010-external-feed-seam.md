# ADR 0010: External-signal feed seam — X posts as logged environment events

Status: accepted (seam + null default only; retrieval provider and experiments deferred to P5+)
Phase: cross-cutting (rides on the P2 Game-Master / tier-2 machinery; interpreted at P5)

## Spike question

Where does "inject some X data as events that certain actors receive (via RAG)"
fit, and what is the minimal stub that respects the architecture?

## Context

The project brief wants a proportion of personas seeded/pressured from scraped DST
opinion posts. This request is the *runtime* variant: external posts delivered to
**specific agents during a run**. The rule set already anticipates it — **R3**:
"any deliberately shared signal (e.g. a news feed) must be logged so its causal
effect on convergence can be traced," and `run-architecture.md` lists "any
deliberately shared signal (news feed)" as tier-2 shared state. A feed is a
convergence *driver*, so it cannot be a hidden input (first-principles #4), and its
effect is only *measurable* once the divergence metric exists (P5).

## Options considered

1. **Environment → agent feed via a logged provider seam (chosen).** A pluggable
   `FeedProvider` (parallel to the `topology` seam) yields `FeedEvent`s per tick;
   the simulation delivers each into the *target* agent's private memory as a
   `KIND_FEED` record and logs a distinct `feed_delivery` event with the post's
   `source`. Default `NullFeedProvider` → zero behaviour change. RAG retrieval is a
   documented `RagFeedProvider` stub.
2. **Route feed through the Game Master** as an action-like effect. Rejected: a feed
   is *environment → agent*, not the *agent → agent* path the GM resolves (R2/R24);
   overloading the GM blurs that boundary and would fold the shared signal into the
   `action`/`memory_write` streams instead of its own.
3. **Seed-time injection only** (pre-load opinionated memories at population build).
   Rejected *as the answer to this request* (it's about runtime events) but noted as
   the sibling seam — the "opinionated-minority slider" from the brief.

## Decision

Option 1. `src/polis/feed.py`: `FeedEvent` (target, text, importance, `source`
provenance, created_at), a `FeedProvider` Protocol, `NullFeedProvider` (default),
`ScriptedFeedProvider` (deterministic, for tests + the from-file shape), and
`RagFeedProvider` (stub raising `NotImplementedError` with the intended design).
`memory.KIND_FEED` and `runlog.EVENT_FEED` added. `Simulation` gains a `feed=`
param; `_deliver_feed` runs at **tick start, before the decide phase**, so an agent
can react to the day's feed the same tick — uniform pre-tick for all agents, hence
independent of the R28 within-tick scheme. `feed_provider` is recorded in the run
config (R3/R17).

## Why

The feed is exactly R3's "deliberately shared signal": modelling it as a logged,
config-versioned environment stream — separate from agent memory-writes — is what
makes its causal contribution to convergence *separable* later, rather than an
untraceable nudge. Keeping it a null-default seam means it ships now without
perturbing any existing run, matching the house pattern (the `executor="batch"` and
`topology` seams, the `Run.metrics` stub).

## Consequences

- **Retrieval (which posts → which actor) is unimplemented** (`RagFeedProvider`).
  It needs the scraped corpus (ToS-sensitive, CLAUDE.md) and — to *interpret* its
  effect — the P5 divergence metric. Use `ScriptedFeedProvider` with a curated
  schedule until then.
- **Delivery timing (before decide) is a deliberate, revisitable choice.** A feed
  read before acting maximises same-tick reactivity; if a "sees it next tick" model
  is wanted, deliver as a resolve-phase effect instead — recorded so it's not an
  accident (cf. R28's rationale for making within-tick ordering explicit).
- **Targeting is partial by design** — only named agents receive events, which is
  the natural home for a "committed/opinionated minority" (R11); *who* is logged so
  its role is traceable.
- Adds `KIND_FEED` / `EVENT_FEED` to the closed vocabularies; `ScriptedFeedProvider`
  is the from-file provider shape. No change to agent/LLM contracts.

## Rules touched

R3 (shared signal logged for traceability — the core rule), R2 (feed is
environment→agent, kept off the agent→agent path), R17 (feed source versioned in
config). Sibling to R11 (committed minority via partial targeting). Interpreted
against R14–R16 at P5.
