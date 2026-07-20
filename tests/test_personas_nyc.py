"""Structural guards for the thick NYC cast (R7) — no network.

Not asserting the *content* of authored personas (that's judgment, not a unit test),
but the invariants a run relies on: the null persona stays thin, thick personas are
actually anchored, ids are unique, and seed specs are well-formed.
"""
from polis.personas_nyc import (
    BY_ID,
    NULL_PERSONA,
    NYC_CAST,
    THICK_PERSONAS,
    null_cast,
)
from polis.persona import Persona
from polis.prompts import persona_system


def test_null_persona_is_thin():
    assert NULL_PERSONA.values == ()
    assert NULL_PERSONA.dispositions == ()
    # Same prompt the thin path has always produced (the R16 baseline must not drift).
    assert NULL_PERSONA.system_prompt() == persona_system(NULL_PERSONA.description)


def test_every_thick_persona_is_actually_anchored():
    for p in THICK_PERSONAS:
        assert p.values, f"{p.id} has no values (R7)"
        assert p.dispositions, f"{p.id} has no dispositions (R7)"
        # Anchors reach the prompt.
        assert p.values[0] in p.system_prompt()


def test_persona_ids_are_unique():
    ids = [p.id for p in THICK_PERSONAS]
    assert len(ids) == len(set(ids))
    assert set(BY_ID) == set(ids)


def test_seed_specs_are_well_formed():
    for ps in NYC_CAST:
        for s in ps.memories:
            assert s.text
            assert 1.0 <= s.importance <= 10.0
            assert s.created_at <= 0.0  # seeded in the past on the abstract axis


def test_null_cast_gives_distinct_thin_ids():
    cast = null_cast(4)
    assert [p.id for p in cast] == ["resident_0", "resident_1", "resident_2", "resident_3"]
    assert all(isinstance(p, Persona) and p.values == () for p in cast)
    assert len({p.id for p in cast}) == 4
