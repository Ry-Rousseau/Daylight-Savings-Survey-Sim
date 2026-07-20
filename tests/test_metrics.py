"""Homogeneity-metric tests (R14) — the P4 measurement instrument.

The pure math and the log-reading logic are network-free; a small hand-built
run object drives ``stance_distribution`` without a live simulation.
"""
from math import log

import pytest

from polis.metrics import homogeneity, homogeneity_trajectory, stance_distribution
from polis.runlog import EVENT_ACTION


def test_consensus_is_maximally_homogeneous():
    h = homogeneity({"keep DST": 5})
    assert h["dominant_share"] == 1.0
    assert h["distinct"] == 1
    assert h["entropy"] == 0.0
    assert h["dominant_stance"] == "keep DST"


def test_even_split_is_maximally_diverse_over_observed():
    h = homogeneity({"a": 5, "b": 5})
    assert h["dominant_share"] == 0.5
    assert h["distinct"] == 2
    assert h["entropy"] == pytest.approx(1.0)  # uniform over the 2 observed stances


def test_support_normalisation_rewards_using_fewer_of_the_options():
    # Even split across 2 of 4 possible options is only half the max entropy.
    h = homogeneity({"a": 5, "b": 5}, support=4)
    assert h["entropy"] == pytest.approx(log(2) / log(4))


def test_plurality_share_and_distinct():
    h = homogeneity({"a": 3, "b": 1})
    assert h["dominant_share"] == 0.75
    assert h["dominant_stance"] == "a"
    assert h["distinct"] == 2


def test_empty_distribution():
    h = homogeneity({})
    assert h == {"n": 0, "dominant_stance": None, "dominant_share": 0.0, "distinct": 0, "entropy": 0.0}


class FakeRun:
    """Minimal stand-in for a ``Run``: an event list + a tick count."""

    def __init__(self, events, ticks):
        self._events = events
        self.ticks = ticks

    def events(self, *, event_type=None):
        return [e for e in self._events if event_type is None or e["event_type"] == event_type]


def _speak(tick, agent_id, stance):
    return {"event_type": EVENT_ACTION, "tick": tick, "agent_id": agent_id,
            "payload": {"action_type": "speak", "stance": stance}}


def _abstain(tick, agent_id):
    return {"event_type": EVENT_ACTION, "tick": tick, "agent_id": agent_id,
            "payload": {"action_type": "abstain"}}


def test_stance_distribution_uses_most_recent_speak():
    run = FakeRun([_speak(0, "a1", "X"), _speak(1, "a1", "Y")], ticks=2)
    assert stance_distribution(run, tick=0) == {"X": 1}   # as-of tick 0
    assert stance_distribution(run) == {"Y": 1}           # latest overwrites


def test_abstain_contributes_no_stance():
    run = FakeRun([_speak(0, "a1", "X"), _abstain(0, "a2")], ticks=1)
    assert stance_distribution(run) == {"X": 1}  # a2 holds no expressed position


def test_trajectory_has_one_row_per_tick():
    run = FakeRun([_speak(0, "a1", "X"), _speak(0, "a2", "X"), _speak(1, "a2", "Y")], ticks=2)
    traj = homogeneity_trajectory(run)
    assert [r["tick"] for r in traj] == [0, 1]
    assert traj[0]["dominant_share"] == 1.0   # both on X at tick 0 (consensus)
    assert traj[1]["distinct"] == 2           # a1 still X, a2 now Y
