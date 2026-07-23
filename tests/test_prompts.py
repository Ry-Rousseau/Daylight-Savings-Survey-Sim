"""Prompt-construction tests — the persistence (anti-sycophancy) clause, no network.

The homogenization surface lives in the prompts (see module docstring), so its opt-in
counter-pressure and its off byte-identity are worth pinning explicitly.
"""
from polis import prompts
from polis.prompts import PERSISTENCE_CLAUSE, action_user

STANCES = ["permanent standard", "permanent dst", "keep switching"]


def test_persistence_off_is_byte_identical():
    """Default off must not change the prompt at all — the existing runs stay reproducible."""
    off = action_user("daylight saving time", STANCES)
    assert PERSISTENCE_CLAUSE not in off


def test_persistence_on_injects_the_clause():
    on = action_user("daylight saving time", STANCES, persistence=True)
    assert PERSISTENCE_CLAUSE in on
    # the only difference from off is the clause
    off = action_user("daylight saving time", STANCES)
    assert on == off.replace(
        "Decide what you do this turn.\n",
        "Decide what you do this turn.\n" + PERSISTENCE_CLAUSE,
    )


def test_persistence_applies_in_deliberate_mode_too():
    on = action_user("dst", STANCES, mode="deliberate", persistence=True)
    assert PERSISTENCE_CLAUSE in on


def test_persistence_default_matches_explicit_false():
    assert action_user("dst", STANCES) == action_user("dst", STANCES, persistence=False)
