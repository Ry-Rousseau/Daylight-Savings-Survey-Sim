"""External-signal feed — deliberately shared events injected into agents (R3).

A ``FeedProvider`` supplies *environment events* (e.g. DST posts scraped from X)
that reach **specific agents** during a run. This is the "news feed" R3 names: a
deliberately shared signal that must be **logged** so its causal effect on
convergence is traceable — a feed is a convergence *driver*, so it can't be a
hidden input.

Boundary (R2/R3): a feed event is an *environment → agent* signal, distinct from
the *agent → agent* path the Game Master resolves. It lands in the **target
agent's private memory** as a ``KIND_FEED`` record, delivered by the simulation's
environment step and logged as its own ``feed_delivery`` stream — never written by
an agent, never folded into the ``action`` stream.

Status: **seam + null default only.** The provider interface, the delivery path,
the memory kind, and the log stream are wired and tested; the actual retrieval
(which posts → which actor) is a stub. Runs use ``NullFeedProvider`` (delivers
nothing) until a real provider is passed, so this changes no existing behaviour.
Experiments that *interpret* a feed's effect wait on the P5 divergence metric —
R3's whole point is being able to measure what the shared signal did.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .world import WorldView


@dataclass(frozen=True)
class FeedEvent:
    """One external post delivered to one agent. ``source`` is the provenance
    handle (e.g. an X post id / url) so the shared signal is traceable back to its
    origin (R3). ``created_at`` defaults to the delivering tick when left None."""

    target_agent_id: str
    text: str
    importance: float = 5.0
    source: str = ""
    created_at: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "effect": "feed_delivery",
            "target_agent_id": self.target_agent_id,
            "text": self.text,
            "importance": self.importance,
            "source": self.source,
            "created_at": self.created_at,
        }


@runtime_checkable
class FeedProvider(Protocol):
    """Yields the external events that reach agents on a given tick.

    The pluggable seam — parallel to the ``topology`` seam — so *which posts reach
    which actors* is a swappable, logged run parameter (R3/R17), never hardcoded.
    Called once per tick, before the decide phase, so an agent can condition on the
    day's feed the same tick.
    """

    def events_for_tick(
        self, tick: int, roster: Sequence[str], world: WorldView
    ) -> list[FeedEvent]:
        ...


class NullFeedProvider:
    """Default: no external signal. Keeps a run a closed system unless a feed is
    deliberately switched on — no hidden shared input (first-principles #4)."""

    def events_for_tick(self, tick, roster, world) -> list[FeedEvent]:
        return []


@dataclass
class ScriptedFeedProvider:
    """A fixed, deterministic feed keyed by tick — for tests and as the worked
    example of the interface. ``schedule`` maps a tick to the events delivered that
    tick; targets not in the roster are dropped by the caller.

    This is also the shape a from-file provider takes: load your curated X posts
    into a ``{tick: [FeedEvent, ...]}`` schedule and pass it in.
    """

    schedule: Mapping[int, Sequence[FeedEvent]] = field(default_factory=dict)

    def events_for_tick(self, tick, roster, world) -> list[FeedEvent]:
        return list(self.schedule.get(tick, ()))


class RagFeedProvider:
    """STUB — retrieval-driven feed: pick which X posts reach which actor.

    Intended design (to implement when the X corpus + the P5 metric are in place):

    1. Embed the scraped DST post corpus once (reuse ``EmbeddingModel``; the local
       4070 Ti is reserved for exactly this lightweight local model).
    2. Per tick, for each *targeted* actor, retrieve the top-k posts most relevant
       to that actor — by persona/interest embedding, or by a matched demographic
       facet once census seeding lands (P5) — and emit them as ``FeedEvent``s with
       the post id as ``source``.
    3. Targeting is deliberately partial: only some actors receive the feed
       (mirrors the "committed/opinionated minority" idea, R11), and *who* is a
       logged run parameter so its causal role is traceable (R3).

    Deferred because: (a) it needs the scraped corpus (ToS-sensitive, CLAUDE.md),
    and (b) its effect is only *readable* once the divergence metric exists (P5).
    """

    def __init__(self, *args, **kwargs):
        self._config = (args, kwargs)

    def events_for_tick(self, tick, roster, world) -> list[FeedEvent]:
        raise NotImplementedError(
            "RagFeedProvider is a stub (P5+). Implement retrieval over the X corpus, "
            "or use ScriptedFeedProvider with a curated schedule for now."
        )
