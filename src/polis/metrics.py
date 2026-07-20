"""Homogeneity metric — the P4-scoped measurement instrument (R14).

Phase 4's spike is *"how much does the divergence metric change under different
topologies?"*, so a **defined-before-the-run, tested** homogeneity measure is the
phase's instrument, not an afterthought — R14 exists precisely to forbid inventing
a metric *after* a suspicious run. This module is deliberately minimal: a
stance-concentration read over the population's SPEAK stances, computed offline
from the durable run log (tier 3).

**P4/P5 boundary.** The full validation layer stays at P5 and *builds on* these
functions — embedding pairwise-distance and cluster count (the rest of R14), the
continuous per-tick dashboard, and the R16 null-model baseline (which only means
something once personas are the controlled variable, i.e. after P5 deepens them).
Nothing here is throwaway; ``homogeneity`` is the shared kernel P5 extends.

**Bounded honestly.** The signal is what agents *say*: an ABSTAIN contributes no
stance that tick (a self-selected sample), and R27 warns a narrow action space can
suppress observable divergence. So a low reading is *stance concentration*, not
proof of internal consensus — the caveat is stated wherever the number is reported.
"""
from __future__ import annotations

from collections import Counter
from math import log
from typing import Any, Mapping

from .actions import ActionType
from .runlog import EVENT_ACTION


def homogeneity(
    distribution: Mapping[str, int], *, support: int | None = None
) -> dict[str, Any]:
    """Summarise how concentrated a stance distribution is.

    ``distribution`` maps stance -> count (e.g. agents holding it). Returns:

    - ``dominant_share`` — plurality fraction (1.0 = unanimous); the headline.
    - ``distinct`` — number of stances actually present.
    - ``entropy`` — normalised Shannon entropy in [0, 1]: 0 = consensus,
      1 = uniform. Normalised by ``log(support)`` when ``support`` (the number of
      *possible* stances, e.g. the option-set size) is given, so consensus onto one
      of K options reads as more homogeneous than an even split across K; falls back
      to ``log(distinct)`` (evenness among the stances that appeared) otherwise.
    """
    counts = Counter({k: v for k, v in dict(distribution).items() if v > 0})
    total = sum(counts.values())
    if total == 0:
        return {"n": 0, "dominant_stance": None, "dominant_share": 0.0, "distinct": 0, "entropy": 0.0}
    distinct = len(counts)
    dominant_stance, dominant_count = counts.most_common(1)[0]
    k = support if support is not None else distinct
    if k <= 1 or distinct <= 1:
        entropy = 0.0
    else:
        h = -sum((c / total) * log(c / total) for c in counts.values())
        entropy = h / log(k)
    return {
        "n": total,
        "dominant_stance": dominant_stance,
        "dominant_share": dominant_count / total,
        "distinct": distinct,
        "entropy": entropy,
    }


def stance_distribution(run, tick: int | None = None) -> Counter:
    """Each agent's most-recent-SPEAK stance as of ``tick`` (latest if ``None``),
    tallied. Read from the ``action`` event stream in append order, so a later SPEAK
    overwrites an earlier one; an agent that has only ever ABSTAINed contributes no
    stance (it holds no expressed position yet)."""
    latest: dict[str, str] = {}
    for e in run.events(event_type=EVENT_ACTION):
        if tick is not None and e["tick"] is not None and e["tick"] > tick:
            continue
        payload = e["payload"]
        if payload.get("action_type") == ActionType.SPEAK.value and payload.get("stance"):
            latest[e["agent_id"]] = payload["stance"]
    return Counter(latest.values())


def homogeneity_trajectory(run, *, support: int | None = None) -> list[dict[str, Any]]:
    """The per-tick homogeneity curve (R15-style trajectory): shows *when* stance
    concentration happens, not just the endpoint. One row per tick of the run."""
    return [
        {"tick": t, **homogeneity(stance_distribution(run, tick=t), support=support)}
        for t in range(run.ticks)
    ]
