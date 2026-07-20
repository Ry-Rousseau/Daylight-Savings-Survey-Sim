"""Seed memories for the Phase 1 DoD demo (Layer 2 — Persona begins).

Same persona for both agents; only the seeded memories differ, so any answer
gap is attributable to memory alone (walking-skeleton discipline: thicken only
the memory axis). ``created_at`` is on the abstract P1 time axis — negative =
further in the past — so recency weighting is demonstrable before the P2 clock.
"""
from __future__ import annotations

from dataclasses import dataclass

from .embeddings import EmbeddingModel
from .memory import KIND_SEED, MemoryRecord, MemoryStore
from .persona import Persona

# One shared, deliberately neutral persona: the memories, not the identity, move.
SHARED_PERSONA = Persona(
    id="resident",
    description="a working New Yorker in your thirties with an ordinary daily routine.",
    temperature=0.8,
)


@dataclass(frozen=True)
class SeedSpec:
    text: str
    importance: float  # authored 1-10 (no LLM call for seeds, ADR 0004)
    created_at: float  # abstract time-units in the past (<= 0)


# Agent A — lived experience valuing long, bright evenings, with the DST
# consequence made explicit (later sunsets) but stopping short of naming the
# survey option, so the model still makes the final inference.
EVENING_SEEDS = [
    SeedSpec("I play in an after-work softball league and games only work when the sun is up past 7pm — later sunsets are everything to me.", 8.0, -2.0),
    SeedSpec("The winter weeks when it's dark by 4:30pm crush me; I never leave the apartment after work and I hate it.", 8.0, -1.0),
    SeedSpec("The best summers of my life were the ones with light late into the evening for dinners on the stoop; I wish every evening could stay bright like that year-round.", 7.0, -5.0),
    SeedSpec("I do all my errands and socializing after work, never in the morning — a dark early evening wastes my whole day.", 6.0, -10.0),
]

# Agent B — lived experience valuing bright mornings, with the DST consequence
# made explicit (permanently later winter sunrises) but not naming the option.
MORNING_SEEDS = [
    SeedSpec("I run at 6am every day, and a pitch-dark morning is miserable and unsafe; I dread anything that would push sunrise even later.", 8.0, -2.0),
    SeedSpec("I walk my kid to school at 7:30am and I'm terrified of them crossing streets in the dark — in midwinter a permanently later sunrise would mean the sun isn't even up yet.", 9.0, -1.0),
    SeedSpec("I read that year-round summer clocks would mean an 8am winter sunrise in New York, and that horrifies me — my whole life happens before noon.", 8.0, -3.0),
    SeedSpec("Dark winter mornings wreck my sleep and my mood; keeping mornings bright matters more to me than anything that happens after dinner.", 7.0, -10.0),
]


def build_store(embedder: EmbeddingModel, specs: list[SeedSpec]) -> MemoryStore:
    """Embed seed specs into a fresh per-agent store."""
    store = MemoryStore()
    for s in specs:
        store.add(
            MemoryRecord(
                text=s.text,
                embedding=embedder.encode(s.text),
                importance=s.importance,
                created_at=s.created_at,
                last_accessed_at=s.created_at,
                kind=KIND_SEED,
            )
        )
    return store
