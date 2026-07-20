"""Phase 0 personas — deliberately thin (Phase 2 anchors them in values, R7)."""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts


@dataclass(frozen=True)
class Persona:
    id: str
    description: str
    temperature: float = 0.8  # per-agent generation param (R1)

    def system_prompt(self) -> str:
        return prompts.persona_system(self.description)


# Phase 0 walking-skeleton cast: three deliberately different New Yorkers.
SEED_PERSONAS = [
    Persona("nurse", "a night-shift ER nurse in the Bronx who relies on morning daylight to wind down after a shift.", 0.7),
    Persona("owner", "the owner of a rooftop bar in Brooklyn whose summer evening trade depends on late sunsets.", 0.9),
    Persona("retiree", "a retired schoolteacher in Queens who finds the twice-a-year clock change disorienting.", 0.8),
]
