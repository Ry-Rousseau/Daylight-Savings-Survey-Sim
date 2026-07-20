"""Shared world state — tier 2 of the state model (R2/R3).

``docs/design/run-architecture.md`` splits state into three tiers: per-agent
private memory (tier 1, ``MemoryStore``), this shared world state (tier 2), and
the durable run log (tier 3, ``RunLog``). The boundary between tiers 1 and 2 is
the R2 separation, enforced *structurally*: agents only ever receive a read-only
``WorldView`` — they hold no reference that can mutate the world. The only writes
to tier 2 come from applying Game-Master-resolved effects (the Simulation applies
them), never directly from an agent.

The ``stance_tally`` is a deliberately shared signal; per R3 every increment is
logged (as a ``world_update`` event) so its causal effect on convergence stays
traceable. Agents do not read the tally in P2 — turning it into a live consensus
pressure is a later, deliberate dynamics choice, not a P2 side effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class WorldView:
    """A read-only snapshot of world state handed to an agent for a decision.

    Frozen, and ``stance_tally`` is a read-only mapping — an agent cannot reach
    back through it to mutate tier-2 state (R2 enforced by construction).
    """

    tick: int
    roster: tuple[str, ...]
    stance_tally: Mapping[str, int]


@dataclass
class WorldState:
    """The mutable tier-2 store. Written only via GM-resolved effects; read by
    agents only through :meth:`view`."""

    roster: tuple[str, ...]
    tick: int = 0
    stance_tally: dict[str, int] = field(default_factory=dict)

    def view(self) -> WorldView:
        """A read-only snapshot for an agent. Copies the tally so a later mutation
        of the store is not visible through an already-handed-out view, and the
        view's mapping cannot write back into the store."""
        return WorldView(
            tick=self.tick,
            roster=self.roster,
            stance_tally=MappingProxyType(dict(self.stance_tally)),
        )

    def record_stance(self, stance: str) -> None:
        """Increment the shared tally for a stance (applied from a WorldUpdate
        effect; the Simulation logs it per R3)."""
        self.stance_tally[stance] = self.stance_tally.get(stance, 0) + 1

    def advance_tick(self) -> None:
        self.tick += 1
