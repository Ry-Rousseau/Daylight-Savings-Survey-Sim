"""Persona-strength / identity-drift probe (R8, Layer 2 — Persona).

R8: *persona strength is measured, not assumed* — periodically probe an agent
against its own baseline to detect identity drift over a run. Thin personas
collapse toward the model's generic "helpful assistant" voice (R7); this module
gives that collapse a number instead of a vibe.

Two readings per probe, both from embeddings of the agent's free-text *reason*
(voice-collapse shows in the prose, not the categorical choice):

- **drift_from_baseline** — cosine distance between the agent's current probe voice
  and its own tick-0 baseline voice. Rising = this agent is moving off its anchor.
- **distance_to_centroid** — cosine distance from the population's mean probe voice.
  Shrinking *across all agents* is the R9 signal: distinct personas converging into
  one collective voice even without an individual cause. Persona diversity is
  necessary but not sufficient, so the two readings are reported side by side — a
  population can hold per-agent anchors yet still collapse collectively.

The probe re-asks a fixed identity question with ``remember=False`` (see
``Agent.answer``) so measuring does not perturb what is measured. The pure geometry
(``cosine_distance`` / ``population_centroid`` / ``centroid_distance``) is
network-free; only ``capture_baseline`` / ``probe_drift`` touch the live agents.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .llm import LLMError
from .survey import SurveyQuestion

# The persona endpoint (Qwen3 on OpenRouter) occasionally returns a schema-invalid
# blob under strict JSON; the sim's decides absorb that via the scheduler's retries,
# but the probe calls the survey path directly, so it retries then skips a flaky
# agent — a diagnostic tolerates a missed probe rather than aborting the whole run.
_PROBE_ATTEMPTS = 3


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cosine similarity, in [0, 2]. 0 = identical direction, 1 = orthogonal,
    2 = opposite. Robust to un-normalised inputs (synthetic test vectors need not be
    unit-norm even though ``EmbeddingModel`` returns normalised vectors)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def population_centroid(vecs: list[np.ndarray]) -> np.ndarray:
    """Mean voice vector of the population at a probe (not re-normalised — cosine
    distance to it is scale-invariant anyway)."""
    return np.mean(np.stack([np.asarray(v, dtype=np.float64) for v in vecs]), axis=0)


def centroid_distance(vec: np.ndarray, centroid: np.ndarray) -> float:
    """Cosine distance of one agent's voice from the population centroid."""
    return cosine_distance(vec, centroid)


@dataclass(frozen=True)
class DriftReading:
    """One agent's drift at one probe (R8). ``distance_to_centroid`` is a snapshot;
    it is the *trajectory* of the population mean across probes that reveals R9
    collective collapse."""

    agent_id: str
    drift_from_baseline: float
    distance_to_centroid: float


def _probe_voice(
    agent, probe: SurveyQuestion, now: float, *, attempts: int = _PROBE_ATTEMPTS
) -> np.ndarray | None:
    """Ask the probe without writeback (remember=False) and embed the reason text —
    the persona's *voice*, where collapse toward a generic tone appears.

    Returns ``None`` if the endpoint fails to yield a schema-valid answer after
    ``attempts`` tries (with backoff), so one flaky probe skips that agent instead of
    aborting the whole measurement."""
    for i in range(attempts):
        try:
            ans = agent.answer(probe, now=now, remember=False)
            return np.asarray(agent.embedder.encode(ans.reason), dtype=np.float64)
        except LLMError:
            if i + 1 < attempts:
                time.sleep(0.5 * (i + 1))
    return None


def capture_baseline(
    agents, probe: SurveyQuestion, *, now: float = 0.0, attempts: int = _PROBE_ATTEMPTS
) -> dict[str, np.ndarray]:
    """Each agent's tick-0 baseline voice on the probe, keyed by persona id — the
    fixed reference every later ``probe_drift`` measures against. Agents whose probe
    fails after ``attempts`` tries are omitted (they simply have no baseline)."""
    out: dict[str, np.ndarray] = {}
    for a in agents:
        voice = _probe_voice(a, probe, now, attempts=attempts)
        if voice is not None:
            out[a.persona.id] = voice
    return out


def probe_drift(
    agents,
    probe: SurveyQuestion,
    baselines: dict[str, np.ndarray],
    *,
    now: float = 0.0,
    attempts: int = _PROBE_ATTEMPTS,
) -> list[DriftReading]:
    """Re-ask the probe and report each agent's drift-from-own-baseline and
    distance-to-population-centroid (R8/R9), in agent order.

    An agent whose probe fails is skipped (no reading); the centroid is taken over the
    agents that did answer. An agent absent from ``baselines`` (joined after the
    baseline capture, or whose baseline probe failed) reports ``drift_from_baseline``
    = 0.0 — no anchor to have moved from yet."""
    voices = []
    for a in agents:
        voice = _probe_voice(a, probe, now, attempts=attempts)
        if voice is not None:
            voices.append((a.persona.id, voice))
    if not voices:
        return []
    centroid = population_centroid([v for _, v in voices])
    readings = []
    for agent_id, voice in voices:
        base = baselines.get(agent_id)
        drift = cosine_distance(base, voice) if base is not None else 0.0
        readings.append(
            DriftReading(
                agent_id=agent_id,
                drift_from_baseline=drift,
                distance_to_centroid=centroid_distance(voice, centroid),
            )
        )
    return readings