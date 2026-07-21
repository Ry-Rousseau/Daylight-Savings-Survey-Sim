"""Divergence / homogeneity metrics — the validation instrument (R14–R17, R27).

R14 forbids inventing a metric *after* a suspicious run, so every measure here is
defined and tested before a live run and reads the durable run log (tier 3) offline.

**Two views of divergence, deliberately kept side by side:**

- *Categorical* (P4 kernel) — ``homogeneity`` over the discrete SPEAK stances:
  dominant-share / distinct / normalised entropy. Answers "how concentrated are the
  positions?" but is blind to *how* they are voiced.
- *Continuous* (P5, R14's pairwise-distance + cluster-count) — ``pairwise_dispersion``
  and ``cluster_count`` over the **SPEAK utterance embeddings**. Catches
  within-stance wording homogenization the categorical read cannot: two agents on the
  same stance whose language collapses into one voice, or the same stance held in
  genuinely different terms.

``divergence_trajectory`` logs both per tick (R15 continuous dashboard);
``divergence_summary`` bundles the endpoint so a null-persona run and a thick run are
comparable (R16). Metric parameters (``support``, cluster ``threshold``) are explicit
so a reading is traceable to how it was computed (R17).

**R27 is a *separate* gate.** ``action_space_adequacy`` is not a homogeneity number:
a narrow action space (SPEAK/ABSTAIN) can cap observable divergence in a way the R16
baseline won't catch, since the ceiling is set before the null comparison. Read it
*before* trusting any convergence figure.

**Bounded honestly.** The signal is what agents *say*: an ABSTAIN contributes no
stance that tick (a self-selected sample). So a low reading is *stance concentration*,
not proof of internal consensus — the caveat is stated wherever a number is reported.
"""
from __future__ import annotations

from collections import Counter
from math import log
from typing import Any, Mapping

import numpy as np

from .actions import ActionType
from .runlog import EVENT_ACTION

# Default cosine-distance threshold for cluster_count. Utterance embeddings (BGE-small,
# unit-normalised) put near-paraphrases within ~0.15 and distinct positions well beyond
# it; the value is deliberately a *parameter*, swept in the 5C notebook and recorded in
# the metric call (R17), not a hidden constant.
DEFAULT_CLUSTER_THRESHOLD = 0.15


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


_STANCE_ACTIONS = {ActionType.SPEAK.value, ActionType.REBUT.value}


def latest_speaks(run, tick: int | None = None) -> dict[str, dict]:
    """Each agent's most-recent *stance-expressing* payload as of ``tick`` (latest if
    ``None``), keyed by agent id. A SPEAK or a REBUT both state a position (a REBUT is
    a SPEAK framed as pushback), so both count; a SHARE_CONSIDERATION carries no stance
    and an ABSTAIN nothing, so an agent that has only done those is absent (it holds no
    expressed position yet). The shared reader behind both the categorical stance read
    and the embedding utterance read."""
    latest: dict[str, dict] = {}
    for e in run.events(event_type=EVENT_ACTION):
        if tick is not None and e["tick"] is not None and e["tick"] > tick:
            continue
        payload = e["payload"]
        if payload.get("action_type") in _STANCE_ACTIONS and payload.get("stance"):
            latest[e["agent_id"]] = payload
    return latest


def stance_distribution(run, tick: int | None = None) -> Counter:
    """Tally of each agent's most-recent SPEAK stance as of ``tick`` (latest if
    ``None``). See :func:`latest_speaks` for the read semantics."""
    return Counter(p["stance"] for p in latest_speaks(run, tick).values())


def latest_utterances(run, tick: int | None = None) -> dict[str, str]:
    """Each agent's most-recent SPEAK *utterance* text as of ``tick``, keyed by agent
    id. The continuous-divergence read operates on these; agents whose latest SPEAK
    carried no utterance are skipped."""
    return {
        aid: p["utterance"]
        for aid, p in latest_speaks(run, tick).items()
        if p.get("utterance")
    }


def homogeneity_trajectory(run, *, support: int | None = None) -> list[dict[str, Any]]:
    """The per-tick homogeneity curve (R15-style trajectory): shows *when* stance
    concentration happens, not just the endpoint. One row per tick of the run."""
    return [
        {"tick": t, **homogeneity(stance_distribution(run, tick=t), support=support)}
        for t in range(run.ticks)
    ]


# --- continuous divergence over utterance embeddings (R14) ---------------------


def _unit_matrix(vectors) -> np.ndarray:
    """Stack vectors into a row-unit-normalised (n, d) matrix. Zero vectors are left
    as zero rows (their cosine similarity to anything is 0)."""
    m = np.asarray(np.stack([np.asarray(v, dtype=np.float64) for v in vectors]))
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return m / norms


def pairwise_dispersion(vectors) -> float:
    """Mean pairwise cosine *distance* (1 - similarity) across the voice vectors —
    the continuous complement to categorical dominant-share. 0 = every voice
    identical in direction; higher = more dispersed. Fewer than two vectors → 0.0
    (no pair to compare)."""
    if len(vectors) < 2:
        return 0.0
    m = _unit_matrix(vectors)
    sims = m @ m.T
    iu = np.triu_indices(len(m), k=1)
    return float(np.mean(1.0 - sims[iu]))


def cluster_count(vectors, *, threshold: float = DEFAULT_CLUSTER_THRESHOLD) -> int:
    """Number of opinion clusters among the voice vectors: single-linkage connected
    components where an edge joins two vectors within ``threshold`` cosine distance.
    0 vectors → 0; one collective voice → 1. ``threshold`` is explicit and tunable —
    cluster count is inherently threshold-dependent, so the value is recorded by the
    caller (R17), never assumed."""
    n = len(vectors)
    if n == 0:
        return 0
    m = _unit_matrix(vectors)
    connected = (1.0 - (m @ m.T)) <= threshold  # boolean adjacency, single-linkage
    seen = [False] * n
    components = 0
    for start in range(n):
        if seen[start]:
            continue
        components += 1
        stack = [start]
        seen[start] = True
        while stack:  # BFS/DFS over the adjacency to mark one whole component
            i = stack.pop()
            for j in range(n):
                if not seen[j] and connected[i, j]:
                    seen[j] = True
                    stack.append(j)
    return components


def _embed(embedder, texts: list[str], cache: dict[str, np.ndarray] | None) -> list[np.ndarray]:
    """Embed ``texts``, reusing ``cache`` (text -> vector) so the trajectory embeds
    each distinct utterance once across all ticks. ``embedder.encode(list)`` returns
    an (n, d) array; a single text encodes to (d,)."""
    if cache is None:
        return list(np.atleast_2d(embedder.encode(list(texts))))
    missing = [t for t in dict.fromkeys(texts) if t not in cache]
    if missing:
        for t, v in zip(missing, np.atleast_2d(embedder.encode(missing))):
            cache[t] = np.asarray(v, dtype=np.float64)
    return [cache[t] for t in texts]


def utterance_divergence(
    run,
    embedder,
    *,
    tick: int | None = None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
    _cache: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """The continuous divergence read at ``tick`` (latest if ``None``): embed each
    agent's most-recent utterance and summarise pairwise dispersion + cluster count
    (R14). ``n`` is how many agents held an utterance to compare."""
    texts = list(latest_utterances(run, tick).values())
    if not texts:
        return {"n": 0, "pairwise_dispersion": 0.0, "cluster_count": 0}
    vecs = _embed(embedder, texts, _cache)
    return {
        "n": len(texts),
        "pairwise_dispersion": pairwise_dispersion(vecs),
        "cluster_count": cluster_count(vecs, threshold=threshold),
    }


def divergence_trajectory(
    run,
    embedder,
    *,
    support: int | None = None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[dict[str, Any]]:
    """The continuous per-tick dashboard (R15): one row per tick carrying *both* the
    categorical stance read (dominant-share / distinct / entropy) and the embedding
    read (pairwise dispersion / cluster count), so *when* convergence happens is
    visible on either axis. Each distinct utterance is embedded once."""
    cache: dict[str, np.ndarray] = {}
    rows = []
    for t in range(run.ticks):
        cat = homogeneity(stance_distribution(run, tick=t), support=support)
        emb = utterance_divergence(run, embedder, tick=t, threshold=threshold, _cache=cache)
        rows.append({
            "tick": t,
            **cat,
            "pairwise_dispersion": emb["pairwise_dispersion"],
            "cluster_count": emb["cluster_count"],
            "n_utterances": emb["n"],
        })
    return rows


def divergence_summary(
    run,
    embedder,
    *,
    support: int | None = None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, Any]:
    """Endpoint metric bundle (latest tick) merging the categorical and embedding
    reads — the unit a notebook computes for a **null-persona** run and a **thick**
    run to compare (R16). The R16 baseline is only interpretable once persona is the
    controlled variable (ADR 0012); this is the instrument that reads both."""
    cat = homogeneity(stance_distribution(run), support=support)
    emb = utterance_divergence(run, embedder, threshold=threshold)
    return {
        **cat,
        "pairwise_dispersion": emb["pairwise_dispersion"],
        "cluster_count": emb["cluster_count"],
        "n_utterances": emb["n"],
    }


# --- R27: action-space adequacy, a SEPARATE diagnostic from homogeneity --------


def action_space_adequacy(
    run,
    *,
    stances: list[str] | None = None,
    embedder=None,
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> dict[str, Any]:
    """Whether the closed SPEAK/ABSTAIN action space (ADR 0008) is actually being
    *exercised* — read **before** any homogeneity number is trusted (R27). A narrow
    or degenerately-used action space caps observable divergence in a way the R16
    null baseline will not catch, because the ceiling is set before that comparison.

    Reports usage across the whole run (every tick, not just the endpoint): abstain
    rate, how many of the available stances were ever used, and how varied the
    utterances are (uniqueness, and optionally embedding dispersion if an ``embedder``
    is passed). ``flags`` lists obvious degeneracies that would cap divergence."""
    n_speak = n_abstain = n_consider = n_rebut = 0
    speaks: list[dict] = []  # stance-expressing actions (SPEAK + REBUT)
    for e in run.events(event_type=EVENT_ACTION):
        p = e["payload"]
        at = p.get("action_type")
        if at == ActionType.SPEAK.value and p.get("stance"):
            n_speak += 1
            speaks.append(p)
        elif at == ActionType.REBUT.value and p.get("stance"):
            n_rebut += 1
            speaks.append(p)  # a rebut states a position, so it counts toward stances/utterances
        elif at == ActionType.ABSTAIN.value:
            n_abstain += 1
        elif at == ActionType.SHARE_CONSIDERATION.value and p.get("consideration"):
            n_consider += 1
    n_actions = n_speak + n_abstain + n_consider + n_rebut
    n_stated = n_speak + n_rebut  # actions that expressed a stance
    stances_used = {p["stance"] for p in speaks}
    utterances = [p["utterance"] for p in speaks if p.get("utterance")]
    distinct_utterances = len(set(utterances))
    out: dict[str, Any] = {
        "n_actions": n_actions,
        "n_speak": n_speak,
        "n_abstain": n_abstain,
        # SHARE_CONSIDERATION usage (R27): a consideration circulates a reason with
        # no stance, so it widens the action space beyond vote-broadcasting; a run
        # exercising it is visible here even though it never moves the stance tally.
        "n_consider": n_consider,
        # REBUT usage (R27): a rebut is a SPEAK framed as pushback — it does state a
        # position (so it counts as stated), but tracking it separately shows how much
        # of the discourse is active disagreement vs plain endorsement.
        "n_rebut": n_rebut,
        "consider_rate": (n_consider / n_actions) if n_actions else 0.0,
        "rebut_rate": (n_rebut / n_actions) if n_actions else 0.0,
        "abstain_rate": (n_abstain / n_actions) if n_actions else 0.0,
        "distinct_stances_used": len(stances_used),
        "distinct_utterances": distinct_utterances,
        "utterance_uniqueness": (distinct_utterances / n_stated) if n_stated else 0.0,
    }
    if stances is not None:
        out["stance_coverage"] = (len(stances_used) / len(stances)) if stances else 0.0
    if embedder is not None and utterances:
        out["utterance_dispersion"] = pairwise_dispersion(_embed(embedder, utterances, {}))
    out["flags"] = _adequacy_flags(out)
    return out


def _adequacy_flags(m: dict[str, Any]) -> list[str]:
    """Obvious action-space degeneracies that cap observable divergence (R27). Each
    is a reason a low homogeneity reading might be an artifact of the action space
    rather than genuine consensus."""
    flags = []
    n_stated = m["n_speak"] + m.get("n_rebut", 0)  # SPEAK + REBUT both state a position
    if m["n_actions"] > 0 and n_stated == 0 and m.get("n_consider", 0) == 0:
        flags.append("all_abstain")  # nothing but abstentions — the space collapsed
    if n_stated > 0 and m["distinct_stances_used"] <= 1:
        flags.append("single_stance")  # positions collapsed to one option
    if n_stated >= 2 and m["utterance_uniqueness"] < 0.5:
        flags.append("low_utterance_variety")  # agents repeating the same wording
    return flags
