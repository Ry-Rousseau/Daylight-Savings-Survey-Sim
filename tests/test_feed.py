"""Feed stub tests — external shared signals injected into targeted agents (R3).

No network: a fake embedder + fixed-action client, and a ScriptedFeedProvider, so
the environment→agent delivery path, the KIND_FEED memory, and the feed_delivery
log stream are exercised deterministically. Default (NullFeedProvider) must change
nothing — proven by the untouched P2/P3 suites plus the null test here.
"""
from dataclasses import dataclass

import numpy as np
import pytest

from polis.agent import Agent
from polis.feed import FeedEvent, NullFeedProvider, RagFeedProvider, ScriptedFeedProvider
from polis.memory import KIND_FEED, MemoryStore
from polis.persona import Persona
from polis.runlog import EVENT_FEED
from polis.simulation import Population, Simulation


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


@dataclass
class _Cfg:
    model: str = "fake-model"
    base_url: str = "https://fake/v1"


class FakeClient:
    config = _Cfg()

    def decide(self, **kw) -> dict:
        return {"action_type": "abstain"}  # abstain: isolate the feed as the only memory source


def _agent(pid: str) -> Agent:
    return Agent(Persona(pid, f"person {pid}"), FakeClient(),
                embedder=FakeEmbedder(), memory=MemoryStore())


def _pop():
    return Population([_agent("a1"), _agent("a2")])


def test_null_feed_delivers_nothing():
    pop = _pop()
    run = Simulation(pop).run(2)
    assert all(len(a.memory) == 0 for a in pop.agents)  # abstain + no feed => empty
    assert run.events(event_type=EVENT_FEED) == []


def test_scripted_feed_reaches_only_the_targeted_agent():
    """A post delivered to a1 lands in a1's store as KIND_FEED and nowhere else (R2/R3)."""
    feed = ScriptedFeedProvider(schedule={
        0: [FeedEvent(target_agent_id="a1", text="permanent DST would ruin my mornings",
                      importance=8.0, source="x:12345")],
    })
    pop = _pop()
    run = Simulation(pop, feed=feed).run(2)
    a1, a2 = pop.by_id["a1"], pop.by_id["a2"]
    assert len(a1.memory) == 1 and a1.memory.records[0].kind == KIND_FEED
    assert "mornings" in a1.memory.records[0].text
    assert len(a2.memory) == 0  # not targeted


def test_feed_delivery_is_logged_with_provenance():
    feed = ScriptedFeedProvider(schedule={
        1: [FeedEvent(target_agent_id="a2", text="later sunsets are worth it", source="x:999")],
    })
    pop = _pop()
    run = Simulation(pop, feed=feed).run(2)
    events = run.events(event_type=EVENT_FEED)
    assert len(events) == 1
    e = events[0]
    assert e["tick"] == 1 and e["agent_id"] == "a2"
    assert e["payload"]["source"] == "x:999"  # traceable back to the origin post (R3)


def test_feed_to_unknown_agent_is_dropped_not_crashed():
    feed = ScriptedFeedProvider(schedule={0: [FeedEvent(target_agent_id="ghost", text="hi")]})
    pop = _pop()
    run = Simulation(pop, feed=feed).run(1)
    assert run.events(event_type=EVENT_FEED) == []
    assert all(len(a.memory) == 0 for a in pop.agents)


def test_feed_provider_is_versioned_in_config():
    feed = ScriptedFeedProvider()
    run = Simulation(_pop(), feed=feed).run(1)
    assert run.config["feed_provider"] == "ScriptedFeedProvider"
    # default run records the null provider
    assert Simulation(_pop()).run(1).config["feed_provider"] == "NullFeedProvider"


def test_rag_provider_is_an_unimplemented_stub():
    with pytest.raises(NotImplementedError, match="stub"):
        RagFeedProvider().events_for_tick(0, ("a1",), None)
