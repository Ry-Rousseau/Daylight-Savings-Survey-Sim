"""Central home for every stage's prompt (Layers 1-2, 5).

Prompts are where homogenization pressure hides: the persona voice, how memory
is surfaced, and how a choice is asked all steer an agent toward or away from a
generic consensus voice. Keeping them in one module makes that surface
auditable and ablatable rather than scattered across call sites.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def persona_system(description: str) -> str:
    """System prompt anchoring the agent to its persona (R7 begins at P5)."""
    return (
        f"You are {description} You live in New York City. "
        "Answer the survey as this person would, in their own voice and interests."
    )


def memory_block(memories: Sequence[str]) -> str:
    """Retrieved memories, formatted for injection. Empty -> no block at all."""
    if not memories:
        return ""
    lines = "\n".join(f"- {m}" for m in memories)
    return (
        "Relevant things you remember — these lived experiences strongly shape "
        f"how you see this question:\n{lines}\n\n"
    )


def survey_user(text: str, options: Iterable[str], *, memories: Sequence[str] = ()) -> str:
    """The survey question, prefixed with any retrieved memories."""
    body = text + "\nChoose exactly one:\n" + "\n".join(f"- {o}" for o in options)
    return memory_block(memories) + body


def action_user(topic: str, stances: Sequence[str], *, memories: Sequence[str] = ()) -> str:
    """The per-tick action decision, prefixed with any retrieved memories.

    Abstaining is offered as a first-class choice (R25): forcing everyone to
    speak every tick is itself a homogenizing pressure, so the prompt makes
    staying quiet a legitimate option rather than a failure.
    """
    body = (
        f"You are talking with your neighbors about {topic}. Decide what you do this turn.\n"
        '- To speak up, set "action_type" to "speak", choose the stance that matches your '
        'view, and write a short, natural "utterance" (one or two sentences) in your own voice.\n'
        '- To stay quiet this turn, set "action_type" to "abstain". Say nothing if you have '
        "nothing you feel like adding — staying quiet is a fine choice.\n"
        "Available stances:\n" + "\n".join(f"- {s}" for s in stances)
    )
    return memory_block(memories) + body


def poignancy(text: str) -> str:
    """Ask the model to rate a memory's importance 1-10 (optional path, task 4)."""
    return (
        "On a scale of 1 to 10, where 1 is purely mundane (e.g. brushing teeth) "
        "and 10 is extremely poignant (e.g. a breakup, a move), rate the likely "
        "poignancy of the following memory. Respond with a single integer only.\n\n"
        f"Memory: {text}\nRating:"
    )
