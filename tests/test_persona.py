"""Persona-schema tests (R7) — value/disposition anchoring.

Network-free: exercises only prompt composition. The load-bearing regression guard
is that an *empty* persona still emits the exact P0–P4 thin prompt, so the thin
persona keeps working as the R16 null-model baseline (5B/5C).
"""
from polis.persona import Persona
from polis.prompts import persona_system

THIN_DESCRIPTION = "a working New Yorker in your thirties with an ordinary daily routine."

# The exact string the engine emitted through P0–P4, pinned here so any drift in the
# empty-persona path (which is the null baseline) fails loudly.
EXPECTED_THIN_PROMPT = (
    "You are a working New Yorker in your thirties with an ordinary daily routine. "
    "You live in New York City. "
    "Answer the survey as this person would, in their own voice and interests."
)


def test_empty_persona_prompt_is_byte_identical_to_p0_p4():
    p = Persona("resident", THIN_DESCRIPTION)
    assert p.system_prompt() == EXPECTED_THIN_PROMPT


def test_empty_values_and_dispositions_match_bare_call():
    assert persona_system(THIN_DESCRIPTION, (), ()) == EXPECTED_THIN_PROMPT


def test_values_appear_in_prompt():
    p = Persona(
        "runner",
        "an early-morning runner in Manhattan.",
        values=("safe, well-lit 6am runs", "keeping a disciplined routine"),
    )
    prompt = p.system_prompt()
    assert "safe, well-lit 6am runs" in prompt
    assert "keeping a disciplined routine" in prompt
    # Values are separated so multiple anchors read as distinct commitments.
    assert "safe, well-lit 6am runs; keeping a disciplined routine" in prompt


def test_dispositions_appear_in_prompt():
    p = Persona(
        "student",
        "a night-owl grad student in Brooklyn.",
        dispositions=("easygoing", "tends to go along with the crowd"),
    )
    prompt = p.system_prompt()
    assert "easygoing; tends to go along with the crowd" in prompt


def test_thick_persona_keeps_the_base_identity_line():
    # Anchors are *added to* the identity, never replace it.
    p = Persona(
        "owner",
        "the owner of a rooftop bar in Brooklyn.",
        values=("late summer sunsets for evening trade",),
        dispositions=("argues their corner",),
    )
    prompt = p.system_prompt()
    assert prompt.startswith("You are the owner of a rooftop bar in Brooklyn. You live in New York City.")
    assert "late summer sunsets for evening trade" in prompt
    assert "argues their corner" in prompt


def test_persona_is_hashable_and_frozen():
    # Frozen + tuple anchors => usable as a dict key / in a set (config dedup, R17).
    p = Persona("a", "desc.", values=("x",), dispositions=("y",))
    assert hash(p) is not None
    assert {p: 1}[p] == 1
