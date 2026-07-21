"""Personas — a persona is an agent's anchored identity (Layer 2).

Through P4 a persona was deliberately thin (`id / description / temperature`) and
diversity came from seeded *memory*, not identity. P5 (R7) adds value/disposition
anchoring: ``values`` (what the person cares about) and ``dispositions`` (how they
hold and voice a view). Both default empty, so the thin persona is unchanged and
doubles as the R16 null-model baseline. The thick NYC cast lives in
``personas_nyc``; the conviction-slider seam (seed-time opinion injection) rides on
top of these fields + opinion seed memories, and is not built here.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import prompts


@dataclass(frozen=True)
class Persona:
    id: str
    description: str
    temperature: float = 0.8  # per-agent generation param (R1)
    # Value/disposition anchors (R7). Tuples so the frozen dataclass stays hashable.
    values: tuple[str, ...] = ()
    dispositions: tuple[str, ...] = ()
    # Where the person lives. Defaults to NYC so P0–P5 personas are byte-identical;
    # the P6 census pipeline sets each persona's real US locale (ADR 0015 USA pivot).
    location: str = "New York City"

    def system_prompt(self) -> str:
        return prompts.persona_system(
            self.description, self.values, self.dispositions, self.location
        )


# Phase 0 walking-skeleton cast: three deliberately different New Yorkers.
SEED_PERSONAS = [
    Persona("nurse", "a night-shift ER nurse in the Bronx who relies on morning daylight to wind down after a shift.", 0.7),
    Persona("owner", "the owner of a rooftop bar in Brooklyn whose summer evening trade depends on late sunsets.", 0.9),
    Persona("retiree", "a retired schoolteacher in Queens who finds the twice-a-year clock change disorienting.", 0.8),
]
