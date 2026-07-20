"""Tick-loop tests — the R28 update scheme and effect/log consistency.

No network, no embedding model: a fake embedder returns a constant unit vector
and a fake client returns a fixed action, so the tick loop, Game Master, world
state, and run log are exercised deterministically. Live behavior is in
notebooks/experiments/phase2_interaction_dod.ipynb.
"""
from dataclasses import dataclass

import numpy as np

from polis.agent import Agent
from polis.memory import KIND_HEARD, MemoryStore
from polis.persona import Persona
from polis.runlog import EVENT_MEMORY_WRITE, EVENT_RETRIEVAL, EVENT_WORLD_UPDATE
from polis.simulation import DynamicsConfig, Population, Simulation

STANCE = "Adopt permanent daylight saving time"


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


@dataclass
class _Cfg:
    model: str = "fake-model"


class FakeClient:
    """Returns a fixed action every decision; no network."""

    def __init__(self, action: dict):
        self.action = action
        self.config = _Cfg()

    def decide(self, **kw) -> dict:
        return dict(self.action)


def _agent(pid: str, action: dict) -> Agent:
    return Agent(
        Persona(pid, f"a person called {pid}"),
        FakeClient(action),
        embedder=FakeEmbedder(),
        memory=MemoryStore(),
    )


def _speak_action() -> dict:
    return {"action_type": "speak", "stance": STANCE, "utterance": "later sunsets please"}


def _two_speakers():
    return Population([_agent("a1", _speak_action()), _agent("a2", _speak_action())])


def test_speak_delivers_to_the_other_agent_only():
    """After a SPEAK tick each agent holds exactly the other's utterance (R2)."""
    pop = _two_speakers()
    Simulation(pop).run(1)
    a1, a2 = pop.by_id["a1"], pop.by_id["a2"]
    assert len(a1.memory) == 1 and a1.memory.records[0].kind == KIND_HEARD
    assert "a2" in a1.memory.records[0].text  # heard from the other, never itself
    assert "a2" not in a2.memory.records[0].text
    assert pop.world.stance_tally == {STANCE: 2}


def test_abstain_writes_nothing():
    pop = Population([_agent("a1", {"action_type": "abstain"}),
                     _agent("a2", {"action_type": "abstain"})])
    run = Simulation(pop).run(1)
    assert len(pop.by_id["a1"].memory) == 0 and len(pop.by_id["a2"].memory) == 0
    assert pop.world.stance_tally == {}
    assert run.events(event_type=EVENT_MEMORY_WRITE) == []


def _a2_provenance_hits(scheme: str) -> int:
    pop = _two_speakers()
    run = Simulation(pop, dynamics=DynamicsConfig(update_scheme=scheme)).run(1)
    prov = [e for e in run.events(event_type=EVENT_RETRIEVAL)
            if e["agent_id"] == "a2" and e["tick"] == 0]
    assert len(prov) == 1
    return len(prov[0]["payload"]["hits"])


def test_sequential_lets_second_agent_hear_first_same_tick():
    # Sequential: a1's SPEAK lands before a2 decides, so a2 retrieves it (R28).
    assert _a2_provenance_hits("sequential") == 1


def test_simultaneous_hides_first_agent_within_tick():
    # Simultaneous: a2 decides from the pre-tick snapshot, before any writes (R28).
    assert _a2_provenance_hits("simultaneous") == 0


def test_log_is_consistent_with_applied_effects():
    """No orphan effects: one memory_write event per delivered memory, one
    world_update per tally increment."""
    pop = _two_speakers()
    run = Simulation(pop).run(1)
    total_memories = sum(len(a.memory) for a in pop.agents)
    assert len(run.events(event_type=EVENT_MEMORY_WRITE)) == total_memories
    assert len(run.events(event_type=EVENT_WORLD_UPDATE)) == sum(pop.world.stance_tally.values())


def test_config_is_versioned():
    pop = _two_speakers()
    run = Simulation(pop, dynamics=DynamicsConfig(update_scheme="sequential")).run(2)
    assert run.config["update_scheme"] == "sequential"
    assert run.config["ticks"] == 2
    assert run.config["topology"] == "fully_connected"
    assert {p["id"] for p in run.config["personas"]} == {"a1", "a2"}
    assert len(run.config_hash) == 64  # sha-256 hex
