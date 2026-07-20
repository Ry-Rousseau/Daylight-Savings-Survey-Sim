"""Persona-strength / drift-probe tests (R8/R9).

The geometry is network-free on synthetic vectors. The orchestration
(``capture_baseline`` / ``probe_drift``) is driven by a tiny fake agent whose voice
is a canned vector per probe, so the wiring is proven without a live model.
"""
import numpy as np
import pytest

from polis.drift import (
    DriftReading,
    capture_baseline,
    centroid_distance,
    cosine_distance,
    population_centroid,
    probe_drift,
)
from polis.llm import LLMError
from polis.survey import SurveyAnswer, SurveyQuestion

PROBE = SurveyQuestion(text="probe", options=["a", "b"])


# --- pure geometry --------------------------------------------------------------

def test_identical_vectors_have_zero_drift():
    v = np.array([0.6, 0.8], dtype=np.float32)
    assert cosine_distance(v, v) == pytest.approx(0.0)


def test_opposite_vectors_are_maximally_distant():
    a = np.array([1.0, 0.0])
    assert cosine_distance(a, -a) == pytest.approx(2.0)


def test_orthogonal_vectors_are_distance_one():
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_cosine_distance_is_scale_invariant():
    a = np.array([1.0, 2.0])
    assert cosine_distance(a, 5.0 * a) == pytest.approx(0.0)


def test_zero_vector_is_safe():
    assert cosine_distance(np.zeros(3), np.array([1.0, 0.0, 0.0])) == 0.0


def test_centroid_is_the_mean():
    c = population_centroid([np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    assert c == pytest.approx(np.array([0.5, 0.5]))


def test_convergence_shrinks_distance_to_centroid():
    # A spread population vs a near-collapsed one: mean distance-to-centroid drops as
    # voices converge (the R9 collective-collapse signal).
    spread = [np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([-1.0, 0.0])]
    tight = [np.array([1.0, 0.0]), np.array([1.0, 0.05]), np.array([1.0, -0.05])]
    spread_c, tight_c = population_centroid(spread), population_centroid(tight)
    spread_mean = np.mean([centroid_distance(v, spread_c) for v in spread])
    tight_mean = np.mean([centroid_distance(v, tight_c) for v in tight])
    assert tight_mean < spread_mean


# --- orchestration over fake agents --------------------------------------------

class _FakeEmbedder:
    """Encodes a text by looking it up in a fixed text->vector table."""

    def __init__(self, table):
        self.table = table

    def encode(self, text):
        return np.asarray(self.table[text], dtype=np.float32)


class _FakeAgent:
    """Returns a scripted reason string per call, cycling through ``reasons``; its
    embedder maps those strings to vectors. No model, no network."""

    def __init__(self, agent_id, reasons, table):
        self.persona = type("P", (), {"id": agent_id})()
        self.embedder = _FakeEmbedder(table)
        self._reasons = list(reasons)
        self._i = 0
        self.remember_calls = []

    def answer(self, question, *, now=0.0, remember=True):
        self.remember_calls.append(remember)
        reason = self._reasons[min(self._i, len(self._reasons) - 1)]
        self._i += 1
        return SurveyAnswer(choice="a", reason=reason)


def test_probe_uses_remember_false():
    table = {"r0": [1.0, 0.0]}
    a = _FakeAgent("x", ["r0"], table)
    capture_baseline([a], PROBE)
    assert a.remember_calls == [False]  # measuring must not write back (R8)


def test_capture_then_probe_measures_drift_from_baseline():
    table = {"base": [1.0, 0.0], "moved": [0.0, 1.0]}
    a = _FakeAgent("x", ["base", "moved"], table)
    baselines = capture_baseline([a], PROBE)
    [reading] = probe_drift([a], PROBE, baselines)
    assert reading.agent_id == "x"
    assert reading.drift_from_baseline == pytest.approx(1.0)  # base ⟂ moved


def test_agent_without_baseline_reports_zero_drift():
    table = {"v": [1.0, 0.0]}
    a = _FakeAgent("newcomer", ["v"], table)
    [reading] = probe_drift([a], PROBE, baselines={})  # not in the baseline set
    assert reading.drift_from_baseline == 0.0


class _FlakyAgent:
    """An agent whose probe always fails (endpoint returns garbage) — used to prove
    the probe skips it instead of aborting."""

    def __init__(self, agent_id):
        self.persona = type("P", (), {"id": agent_id})()
        self.embedder = _FakeEmbedder({})

    def answer(self, question, *, now=0.0, remember=True):
        raise LLMError("no valid survey_answer")


def test_flaky_agent_is_skipped_from_baseline_and_drift():
    table = {"ok": [1.0, 0.0]}
    good = _FakeAgent("good", ["ok"], table)
    bad = _FlakyAgent("bad")
    baselines = capture_baseline([good, bad], PROBE, attempts=1)
    assert set(baselines) == {"good"}  # the flaky agent has no baseline
    readings = probe_drift([good, bad], PROBE, baselines, attempts=1)
    assert [r.agent_id for r in readings] == ["good"]  # and no reading


def test_probe_drift_preserves_agent_order_and_centroid():
    table = {"aa": [1.0, 0.0], "bb": [0.0, 1.0]}
    a = _FakeAgent("a", ["aa"], table)
    b = _FakeAgent("b", ["bb"], table)
    readings = probe_drift([a, b], PROBE, baselines={})
    assert [r.agent_id for r in readings] == ["a", "b"]
    # Both sit symmetrically about the centroid (0.5, 0.5) -> equal centroid distance.
    assert readings[0].distance_to_centroid == pytest.approx(readings[1].distance_to_centroid)
    assert isinstance(readings[0], DriftReading)
