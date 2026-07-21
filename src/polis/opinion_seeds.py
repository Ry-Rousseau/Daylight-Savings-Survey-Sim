"""Opinion seeding — the conviction layer (Layer 2/3; ADR 0010 sibling).

The P6b baseline showed a disposition-rich but *opinion-blank* population collapses to
the model's prior (unanimity + one voice). This module injects **real, pre-existing DST
opinions** — scraped + stance-labelled X posts (``data/processed/tweets_labeled.csv``) —
into a fraction of personas at **seed time**, so some agents enter the run already
holding a position. It is the *seed-time* sibling of the runtime feed (``feed.py``): the
feed delivers outside opinions *during* a run; this pre-loads them at t=0.

Everything is a knob (no hidden assumptions): the assignment **scheme** (polarized
two-camp / random / none), the **phrasing** (how a tweet becomes a memory), the seeded
**fraction** per camp, the number/strength of opinions, and whether a camp is a
**committed** minority (R11, immovable). An ``OpinionPlan`` is a config a run cites (R17);
``apply_opinion_plan`` returns the assignment for provenance.
"""
from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .memory import KIND_SEED, MemoryRecord

_URL = re.compile(r"https?://\S+")
_WS = re.compile(r"\s+")

DEFAULT_TWEETS_PATH = "data/processed/tweets_labeled.csv"


def clean_text(text: str) -> str:
    """Strip URLs and collapse whitespace; keep the authentic voice (mentions, emoji)."""
    return _WS.sub(" ", _URL.sub("", str(text))).strip()


def load_opinion_corpus(
    path: str = DEFAULT_TWEETS_PATH, *, on_topic_only: bool = True, min_chars: int = 15
) -> dict[str, list[dict]]:
    """Group labelled tweets by stance → list of ``{text, reason, likes}`` (cleaned).

    ``on_topic_only`` keeps the on-topic subset; ``min_chars`` drops fragments. Stances
    are the exact ``DST_OPTIONS`` strings (the labels align), so a camp keys straight off
    an option. Sorted by like_count desc so a sampler can prefer higher-signal posts."""
    import pandas as pd

    df = pd.read_csv(path)
    if on_topic_only and "on_topic" in df.columns:
        df = df[df["on_topic"] == True]  # noqa: E712 - pandas mask
    corpus: dict[str, list[dict]] = {}
    for stance, sub in df.dropna(subset=["stance"]).groupby("stance"):
        items = []
        for _, r in sub.sort_values("like_count", ascending=False, na_position="last").iterrows():
            text = clean_text(r["text"])
            if len(text) < min_chars:
                continue
            items.append({
                "text": text,
                "reason": clean_text(r["stance_reason"]) if "stance_reason" in r and r["stance_reason"] else "",
                "likes": int(r["like_count"]) if r.get("like_count") == r.get("like_count") else 0,
            })
        if items:
            corpus[str(stance)] = items
    return corpus


# --- phrasing: a tweet → an opinion memory --------------------------------------

def render_opinion(item: Mapping[str, str], phrasing: str) -> str:
    """Turn a corpus item into a memory string.

    - ``seen`` — the raw post as *external exposure* ("I saw someone post…"): authentic,
      a weak anchor (closest to the runtime feed's semantics). The P6 default.
    - ``conviction`` — the post reframed as the persona's *own held view* (strong anchor).
    - ``reason`` — the LLM-extracted ``stance_reason``. **Note:** this is *analytic*
      third-person commentary ("The tweet expresses…"), not a first-person voice — use
      only when a cleaned argument is wanted over authentic phrasing.
    """
    text = item["text"]
    if phrasing == "seen":
        return f'I recently saw someone post online about daylight saving time: "{text}"'
    if phrasing == "conviction":
        return f"When it comes to daylight saving time, this is how I honestly feel: {text}"
    if phrasing == "reason":
        return f"My take on daylight saving time: {item.get('reason') or text}"
    raise ValueError(f"unknown phrasing {phrasing!r} (seen|conviction|reason)")


# --- the plan -------------------------------------------------------------------

@dataclass(frozen=True)
class OpinionPlan:
    """A seeding configuration a run cites (R17). ``scheme='none'`` = the blank-slate
    baseline (no change). For ``two_camp``, ``camps`` is ``[(stance, fraction), …]``
    (fractions of the roster); ``random`` seeds ``fraction`` of agents with a random
    stance drawn from ``stances`` (or all corpus stances). ``committed`` stances become
    immovable factions (R11)."""

    scheme: str = "none"  # none | two_camp | random
    camps: tuple[tuple[str, float], ...] = ()
    fraction: float = 0.5           # for scheme='random'
    stances: tuple[str, ...] = ()   # candidate stances for scheme='random' (empty = all)
    phrasing: str = "seen"
    n_opinions: int = 2
    importance: float = 7.0
    created_at: float = -2.0        # a prior, just before sim start
    committed: tuple[str, ...] = ()
    seed: int = 42

    def to_config(self) -> dict:
        return {
            "scheme": self.scheme, "camps": [list(c) for c in self.camps],
            "fraction": self.fraction, "stances": list(self.stances),
            "phrasing": self.phrasing, "n_opinions": self.n_opinions,
            "importance": self.importance, "committed": list(self.committed), "seed": self.seed,
        }


def assign_stances(roster: Sequence[str], plan: OpinionPlan) -> dict[str, str]:
    """Deterministically map a subset of agent ids → the stance they're seeded with.
    Unseeded agents are absent. Reproducible from ``plan.seed`` + the roster order."""
    if plan.scheme == "none":
        return {}
    rng = random.Random(f"{plan.seed}-{plan.scheme}")
    order = list(roster)
    rng.shuffle(order)
    out: dict[str, str] = {}
    if plan.scheme == "two_camp":
        i = 0
        for stance, frac in plan.camps:
            k = round(frac * len(order))
            for aid in order[i:i + k]:
                out[aid] = stance
            i += k
    elif plan.scheme == "random":
        stances = list(plan.stances) or None
        k = round(plan.fraction * len(order))
        for aid in order[:k]:
            pool = stances if stances else None
            out[aid] = rng.choice(pool) if pool else aid  # pool required; validated in apply
    else:
        raise ValueError(f"unknown scheme {plan.scheme!r} (none|two_camp|random)")
    return out


def apply_opinion_plan(population, corpus: Mapping[str, list[dict]], plan: OpinionPlan) -> dict:
    """Seed opinion memories into the assigned agents' private stores at t=0, and mark
    committed factions (R11). Returns a provenance dict: the per-agent assignment + the
    plan config. ``population`` supplies the embedder via its agents. Idempotent per call;
    call once on a fresh population."""
    if plan.scheme == "none":
        return {"plan": plan.to_config(), "assignment": {}, "n_seeded": 0}

    assignment = assign_stances(population.roster, plan)
    rng = random.Random(f"{plan.seed}-opinions")
    committed = set(plan.committed)
    for aid, stance in assignment.items():
        agent = population.by_id[aid]
        pool = corpus.get(stance, [])
        if pool:
            picks = rng.sample(pool, min(plan.n_opinions, len(pool)))
            for item in picks:
                text = render_opinion(item, plan.phrasing)
                agent.memory.add(MemoryRecord(
                    text=text,
                    embedding=agent.embedder.encode(text),
                    importance=plan.importance,
                    created_at=plan.created_at,
                    last_accessed_at=plan.created_at,
                    kind=KIND_SEED,
                ))
        if stance in committed:
            agent.committed_stance = stance  # R11: immovable, decides without a model call
    return {
        "plan": plan.to_config(),
        "assignment": assignment,
        "n_seeded": len(assignment),
        "n_committed": sum(1 for s in assignment.values() if s in committed),
    }
