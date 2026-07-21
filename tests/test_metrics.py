"""Divergence-metric tests (R14–R17, R27) — the validation instrument.

The pure math and the log-reading logic are network-free; a small hand-built run
object drives the readers without a live simulation, and a table-backed fake
embedder stands in for BGE so the embedding metrics need no model.
"""
from math import log

import numpy as np
import pytest

from polis.metrics import (
    action_space_adequacy,
    cluster_count,
    divergence_summary,
    divergence_trajectory,
    homogeneity,
    homogeneity_trajectory,
    latest_utterances,
    pairwise_dispersion,
    stance_distribution,
    utterance_divergence,
)
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


def _speak_u(tick, agent_id, stance, utterance):
    return {"event_type": EVENT_ACTION, "tick": tick, "agent_id": agent_id,
            "payload": {"action_type": "speak", "stance": stance, "utterance": utterance}}


def _consider(tick, agent_id, consideration):
    return {"event_type": EVENT_ACTION, "tick": tick, "agent_id": agent_id,
            "payload": {"action_type": "share_consideration", "consideration": consideration}}


class _TableEmbedder:
    """Maps a text to a fixed vector via a lookup table; handles str or list like the
    real EmbeddingModel (single -> (d,), list -> (n, d))."""

    def __init__(self, table):
        self.table = table

    def encode(self, texts):
        if isinstance(texts, str):
            return np.asarray(self.table[texts], dtype=np.float64)
        return np.asarray([self.table[t] for t in texts], dtype=np.float64)


def test_stance_distribution_uses_most_recent_speak():
    run = FakeRun([_speak(0, "a1", "X"), _speak(1, "a1", "Y")], ticks=2)
    assert stance_distribution(run, tick=0) == {"X": 1}   # as-of tick 0
    assert stance_distribution(run) == {"Y": 1}           # latest overwrites


def test_abstain_contributes_no_stance():
    run = FakeRun([_speak(0, "a1", "X"), _abstain(0, "a2")], ticks=1)
    assert stance_distribution(run) == {"X": 1}  # a2 holds no expressed position


def test_consideration_contributes_no_stance_or_utterance():
    # A SHARE_CONSIDERATION carries a reason but no stance, so the categorical stance
    # read and the utterance read must both exclude it, exactly as they exclude an
    # abstain — only the SPEAK counts (ADR 0017).
    run = FakeRun([_speak_u(0, "a1", "X", "later light"), _consider(0, "a2", "my evenings matter")], ticks=1)
    assert stance_distribution(run) == {"X": 1}
    assert latest_utterances(run) == {"a1": "later light"}


def test_trajectory_has_one_row_per_tick():
    run = FakeRun([_speak(0, "a1", "X"), _speak(0, "a2", "X"), _speak(1, "a2", "Y")], ticks=2)
    traj = homogeneity_trajectory(run)
    assert [r["tick"] for r in traj] == [0, 1]
    assert traj[0]["dominant_share"] == 1.0   # both on X at tick 0 (consensus)
    assert traj[1]["distinct"] == 2           # a1 still X, a2 now Y


# --- continuous embedding divergence (R14) -------------------------------------

def test_pairwise_dispersion_zero_for_identical():
    v = [1.0, 0.0]
    assert pairwise_dispersion([v, v, v]) == pytest.approx(0.0)


def test_pairwise_dispersion_one_for_orthogonal_pair():
    assert pairwise_dispersion([[1.0, 0.0], [0.0, 1.0]]) == pytest.approx(1.0)


def test_pairwise_dispersion_rises_with_spread():
    tight = pairwise_dispersion([[1.0, 0.0], [1.0, 0.05], [1.0, -0.05]])
    wide = pairwise_dispersion([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert wide > tight


def test_pairwise_dispersion_needs_two():
    assert pairwise_dispersion([[1.0, 0.0]]) == 0.0
    assert pairwise_dispersion([]) == 0.0


def test_cluster_count_empty_and_singleton():
    assert cluster_count([]) == 0
    assert cluster_count([[1.0, 0.0]]) == 1


def test_cluster_count_collapses_identical_to_one():
    v = [1.0, 0.0]
    assert cluster_count([v, v, v]) == 1


def test_cluster_count_finds_two_groups():
    # Two tight groups far apart (orthogonal): two clusters at a tight threshold.
    vecs = [[1.0, 0.0], [1.0, 0.02], [0.0, 1.0], [0.02, 1.0]]
    assert cluster_count(vecs, threshold=0.1) == 2


def test_cluster_count_threshold_merges_at_high_threshold():
    vecs = [[1.0, 0.0], [0.0, 1.0]]  # orthogonal -> distance 1.0
    assert cluster_count(vecs, threshold=0.1) == 2   # far apart
    assert cluster_count(vecs, threshold=1.0) == 1   # threshold swallows the gap


# --- utterance readers + divergence over the log -------------------------------

def test_latest_utterances_tracks_most_recent_speak():
    run = FakeRun([_speak_u(0, "a1", "X", "early"), _speak_u(1, "a1", "X", "late")], ticks=2)
    assert latest_utterances(run, tick=0) == {"a1": "early"}
    assert latest_utterances(run) == {"a1": "late"}


def test_latest_utterances_skips_utteranceless_speak():
    # A P4-style SPEAK logged without an utterance contributes a stance but no text.
    run = FakeRun([_speak(0, "a1", "X"), _speak_u(0, "a2", "X", "hi")], ticks=1)
    assert latest_utterances(run) == {"a2": "hi"}
    assert stance_distribution(run) == {"X": 2}  # stance read unchanged by the refactor


def test_utterance_divergence_reads_and_measures():
    table = {"aa": [1.0, 0.0], "bb": [0.0, 1.0]}
    run = FakeRun([_speak_u(0, "a1", "X", "aa"), _speak_u(0, "a2", "Y", "bb")], ticks=1)
    d = utterance_divergence(run, _TableEmbedder(table))
    assert d["n"] == 2
    assert d["pairwise_dispersion"] == pytest.approx(1.0)
    assert d["cluster_count"] == 2


def test_utterance_divergence_empty_when_no_utterances():
    run = FakeRun([_abstain(0, "a1")], ticks=1)
    assert utterance_divergence(run, _TableEmbedder({})) == {
        "n": 0, "pairwise_dispersion": 0.0, "cluster_count": 0}


def test_divergence_trajectory_carries_both_axes():
    table = {"aa": [1.0, 0.0], "bb": [0.0, 1.0]}
    run = FakeRun(
        [_speak_u(0, "a1", "X", "aa"), _speak_u(0, "a2", "X", "aa"),   # tick 0: consensus, same words
         _speak_u(1, "a2", "Y", "bb")],                                # tick 1: a2 diverges
        ticks=2,
    )
    traj = divergence_trajectory(run, _TableEmbedder(table))
    assert [r["tick"] for r in traj] == [0, 1]
    # tick 0: both categorical and embedding say consensus
    assert traj[0]["dominant_share"] == 1.0
    assert traj[0]["pairwise_dispersion"] == pytest.approx(0.0)
    assert traj[0]["cluster_count"] == 1
    # tick 1: both axes show the split
    assert traj[1]["distinct"] == 2
    assert traj[1]["pairwise_dispersion"] == pytest.approx(1.0)
    assert traj[1]["cluster_count"] == 2


def test_divergence_summary_is_the_endpoint_bundle():
    table = {"aa": [1.0, 0.0], "bb": [0.0, 1.0]}
    run = FakeRun([_speak_u(0, "a1", "X", "aa"), _speak_u(0, "a2", "Y", "bb")], ticks=1)
    s = divergence_summary(run, _TableEmbedder(table), support=4)
    assert s["dominant_share"] == 0.5
    assert s["cluster_count"] == 2
    assert s["n_utterances"] == 2


# --- R27 action-space adequacy -------------------------------------------------

def test_adequacy_counts_and_rates():
    run = FakeRun(
        [_speak_u(0, "a1", "X", "u1"), _speak_u(0, "a2", "Y", "u2"), _abstain(0, "a3")],
        ticks=1,
    )
    m = action_space_adequacy(run, stances=["X", "Y", "Z", "W"])
    assert m["n_actions"] == 3
    assert m["n_speak"] == 2
    assert m["n_abstain"] == 1
    assert m["abstain_rate"] == pytest.approx(1 / 3)
    assert m["distinct_stances_used"] == 2
    assert m["stance_coverage"] == pytest.approx(0.5)
    assert m["utterance_uniqueness"] == 1.0
    assert m["flags"] == []


def test_adequacy_flags_all_abstain():
    run = FakeRun([_abstain(0, "a1"), _abstain(0, "a2")], ticks=1)
    m = action_space_adequacy(run)
    assert m["abstain_rate"] == 1.0
    assert "all_abstain" in m["flags"]


def test_adequacy_flags_single_stance():
    run = FakeRun([_speak_u(0, "a1", "X", "u1"), _speak_u(0, "a2", "X", "u2")], ticks=1)
    m = action_space_adequacy(run)
    assert "single_stance" in m["flags"]  # positions collapsed to one option


def test_adequacy_flags_low_utterance_variety():
    # Same wording parroted -> the space isn't exercising its range (R27).
    run = FakeRun(
        [_speak_u(0, "a1", "X", "same"), _speak_u(0, "a2", "Y", "same"),
         _speak_u(0, "a3", "Z", "same")],
        ticks=1,
    )
    m = action_space_adequacy(run)
    assert m["utterance_uniqueness"] == pytest.approx(1 / 3)
    assert "low_utterance_variety" in m["flags"]


def test_adequacy_optional_embedding_dispersion():
    table = {"u1": [1.0, 0.0], "u2": [0.0, 1.0]}
    run = FakeRun([_speak_u(0, "a1", "X", "u1"), _speak_u(0, "a2", "Y", "u2")], ticks=1)
    m = action_space_adequacy(run, embedder=_TableEmbedder(table))
    assert m["utterance_dispersion"] == pytest.approx(1.0)


def test_adequacy_reports_consideration_usage():
    # R27: a run that circulates reasons shows up as consideration usage, and those
    # actions count toward n_actions without inflating the stance/utterance reads.
    run = FakeRun(
        [_speak_u(0, "a1", "X", "u1"), _consider(0, "a2", "my evenings matter"),
         _consider(1, "a3", "I drive at dawn")],
        ticks=2,
    )
    m = action_space_adequacy(run)
    assert m["n_consider"] == 2
    assert m["n_speak"] == 1 and m["n_abstain"] == 0
    assert m["n_actions"] == 3
    assert m["consider_rate"] == pytest.approx(2 / 3)


def test_adequacy_all_considerations_not_flagged_all_abstain():
    # An all-consideration run expresses no stance but is NOT a degenerate space —
    # it is exercising the new action, so all_abstain must not fire (ADR 0017).
    run = FakeRun([_consider(0, "a1", "reason one"), _consider(0, "a2", "reason two")], ticks=1)
    m = action_space_adequacy(run)
    assert m["n_consider"] == 2 and m["n_speak"] == 0
    assert "all_abstain" not in m["flags"]
