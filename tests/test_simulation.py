"""Tick-loop tests — the R28 update scheme and effect/log consistency.

No network, no embedding model: a fake embedder returns a constant unit vector
and a fake client returns a fixed action, so the tick loop, Game Master, world
state, and run log are exercised deterministically. Live behavior is in
notebooks/experiments/phase2_interaction_dod.ipynb.
"""
from dataclasses import dataclass

import numpy as np

import pytest

from polis.agent import Agent
from polis.memory import KIND_HEARD, MemoryStore
from polis.persona import Persona
from polis.runlog import (
    EVENT_ACTION,
    EVENT_MEMORY_WRITE,
    EVENT_RETRIEVAL,
    EVENT_TICK_METRICS,
    EVENT_WORLD_UPDATE,
)
from polis.questions import DST_OPTIONS
from polis.scheduler import SchedulerConfig
from polis.simulation import DynamicsConfig, Population, Simulation
from polis.topology import RingLattice

STANCE = "Adopt permanent daylight saving time"


class FakeEmbedder:
    def encode(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


@dataclass
class _CfgWithUrl:
    model: str = "fake-model"
    base_url: str = "https://fake.endpoint/v1"


class FakeClient:
    """Returns a fixed action every decision; no network. ``usage`` lets a test
    exercise the token-accounting path; ``fail_times`` makes the first N calls
    raise so the scheduler's retry path is exercised deterministically."""

    def __init__(self, action: dict, *, usage: dict | None = None, fail_times: int = 0):
        self.action = action
        self.usage = usage
        self.config = _CfgWithUrl()
        self._fails_left = fail_times

    def decide(self, **kw) -> dict:
        if self._fails_left > 0:
            self._fails_left -= 1
            raise RuntimeError("429 rate limited")
        out = dict(self.action)
        out["model"] = self.config.model
        out["usage"] = self.usage
        return out


def _agent(pid: str, action: dict, **kw) -> Agent:
    return Agent(
        Persona(pid, f"a person called {pid}"),
        FakeClient(action, **kw),
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


# --- Phase 3: scheduling & throughput ----------------------------------------


def test_concurrent_decide_preserves_effect_and_log_consistency():
    """The serial-writer invariant holds under the concurrent decide phase: with
    10 agents fanned out at concurrency 4, every SPEAK still lands once and the log
    has no orphan effects (all log writes stay on the main thread)."""
    agents = [_agent(f"a{i}", _speak_action()) for i in range(10)]
    pop = Population(agents)
    run = Simulation(pop, scheduler_config=SchedulerConfig(max_concurrency=4)).run(2)
    # Fully-connected, 10 agents, both ticks: each agent hears the other 9 each tick.
    total_memories = sum(len(a.memory) for a in pop.agents)
    assert total_memories == 10 * 9 * 2
    assert len(run.events(event_type=EVENT_MEMORY_WRITE)) == total_memories
    assert len(run.events(event_type=EVENT_WORLD_UPDATE)) == sum(pop.world.stance_tally.values())


def test_action_events_carry_latency_and_tokens():
    pop = Population([_agent("a1", _speak_action(), usage={
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120})])
    run = Simulation(pop).run(1)
    action_events = run.events(event_type=EVENT_ACTION)
    assert len(action_events) == 1
    p = action_events[0]["payload"]
    assert p["total_tokens"] == 120 and p["prompt_tokens"] == 100
    assert p["model"] == "fake-model"  # R6: model pinned per call
    assert p["latency_s"] >= 0 and p["attempts"] == 1


def test_tick_metrics_logged_per_tick():
    pop = _two_speakers()
    run = Simulation(pop).run(3)
    tms = run.events(event_type=EVENT_TICK_METRICS)
    assert [e["payload"]["tick"] for e in tms] == [0, 1, 2]  # one per tick (R15 trajectory)
    assert all(e["payload"]["n_calls"] == 2 for e in tms)


def test_run_throughput_aggregate():
    pop = Population([_agent(f"a{i}", _speak_action(), usage={
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}) for i in range(3)])
    run = Simulation(pop, scheduler_config=SchedulerConfig(
        max_concurrency=3, price_per_mtok=(1.0, 2.0))).run(2)
    t = run.throughput
    assert t["n_agents"] == 3 and t["n_calls"] == 6
    assert t["total_tokens"] == 6 * 15
    assert t["prompt_tokens"] == 60 and t["completion_tokens"] == 30
    # cost = (60 * $1 + 30 * $2) / 1e6
    assert t["est_cost_usd"] == pytest.approx((60 * 1.0 + 30 * 2.0) / 1_000_000)
    assert t["decides_per_s"] > 0 and t["max_concurrency"] == 3


def test_provider_and_scheduler_pinned_in_config():
    """R6/R17: provider (base_url + model) and the throughput knobs are versioned."""
    pop = _two_speakers()
    run = Simulation(pop, scheduler_config=SchedulerConfig(max_concurrency=5)).run(1)
    assert run.config["provider"]["base_url"] == "https://fake.endpoint/v1"
    assert run.config["provider"]["model"] == "fake-model"
    assert run.config["scheduler"]["max_concurrency"] == 5
    assert run.config["scheduler"]["executor"] == "concurrent"


def test_transient_failure_is_retried_then_succeeds():
    # First call raises, scheduler retries; the run completes and records the retry.
    pop = Population([_agent("a1", _speak_action(), fail_times=1,
                            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})])
    run = Simulation(pop, scheduler_config=SchedulerConfig(backoff_base=0.0)).run(1)
    assert run.throughput["retries"] == 1 and run.throughput["failures"] == 0
    assert run.events(event_type=EVENT_ACTION)[0]["payload"]["attempts"] == 2


def test_unrecoverable_failure_aborts_the_run():
    # Fails more times than retries allow → the run raises rather than silently
    # abstaining (which would distort dynamics).
    pop = Population([_agent("a1", _speak_action(), fail_times=99)])
    with pytest.raises(RuntimeError, match="decide failed"):
        Simulation(pop, scheduler_config=SchedulerConfig(max_retries=1, backoff_base=0.0)).run(1)


def test_sequential_scheme_also_logs_throughput():
    pop = _two_speakers()
    run = Simulation(pop, dynamics=DynamicsConfig(update_scheme="sequential")).run(1)
    assert run.throughput["n_calls"] == 2
    assert len(run.events(event_type=EVENT_TICK_METRICS)) == 1


# --- Phase 4: topology / dynamics --------------------------------------------


class RaisingClient:
    """A client whose decide() must never be called (proves a committed agent
    decides without the model). Carries a config so run-config building works."""

    def __init__(self):
        self.config = _CfgWithUrl()

    def decide(self, **kw):  # pragma: no cover - asserted never reached
        raise AssertionError("committed agent must not call the model")


def _committed_agent(pid: str, stance: str) -> Agent:
    return Agent(Persona(pid, f"a person called {pid}"), RaisingClient(),
                 committed_stance=stance, embedder=FakeEmbedder(), memory=MemoryStore())


def test_committed_agent_speaks_fixed_stance_without_model_call():
    """R11: a committed agent SPEAKs its stance deterministically, no client call."""
    other = _agent("a2", _speak_action())
    pop = Population([_committed_agent("a1", STANCE), other])
    run = Simulation(pop).run(1)
    a1_action = [e for e in run.events(event_type=EVENT_ACTION) if e["agent_id"] == "a1"][0]
    assert a1_action["payload"]["action_type"] == "speak"
    assert a1_action["payload"]["stance"] == STANCE
    # It still reaches its neighbour, and the committed roster is versioned (R17).
    assert "a1" in pop.by_id["a2"].memory.records[0].text
    assert run.config["committed"] == [{"id": "a1", "stance": STANCE}]


def test_topology_determines_reach():
    """A sparse graph delivers a SPEAK to exactly its neighbours, not everyone."""
    agents = [_agent(f"a{i}", _speak_action()) for i in range(5)]
    pop = Population(agents)
    run = Simulation(pop, topology=RingLattice(k=2)).run(1)
    # Ring of 5, degree 2: each speaker delivers to 2 listeners → 10 memory writes.
    assert len(run.events(event_type=EVENT_MEMORY_WRITE)) == 5 * 2
    assert run.config["topology"] == {"name": "ring_lattice", "k": 2}


def test_exchange_volume_caps_reach():
    """R12: with a full graph but exchange_volume=1, each SPEAK reaches one listener."""
    agents = [_agent(f"a{i}", _speak_action()) for i in range(5)]
    pop = Population(agents)
    run = Simulation(pop, dynamics=DynamicsConfig(exchange_volume=1, seed=1)).run(1)
    assert len(run.events(event_type=EVENT_MEMORY_WRITE)) == 5 * 1
    assert run.config["exchange_volume"] == 1


def test_homogeneity_reads_from_a_real_run_log():
    """End-to-end wiring: metrics read a real Simulation's SQLite log. Committed
    agents give a deterministic stance split, so the homogeneity read is exact.
    (Topology's *effect* on homogeneity is the live notebook's job — it needs LLM
    agents that update on what they hear; committed agents by design do not.)"""
    from polis.metrics import homogeneity, homogeneity_trajectory, stance_distribution

    other = "Keep the current clock-change"
    pop = Population([_committed_agent(f"a{i}", STANCE if i < 3 else other) for i in range(5)])
    run = Simulation(pop, topology=RingLattice(k=2)).run(2)

    dist = stance_distribution(run)
    assert dist == {STANCE: 3, other: 2}
    h = homogeneity(dist, support=len(DST_OPTIONS))
    assert h["dominant_share"] == 0.6 and h["distinct"] == 2
    assert len(homogeneity_trajectory(run)) == 2  # one row per tick (R15 trajectory)
