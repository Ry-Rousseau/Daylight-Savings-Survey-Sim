"""Central home for every stage's prompt (Layers 1-2, 5).

Prompts are where homogenization pressure hides: the persona voice, how memory
is surfaced, and how a choice is asked all steer an agent toward or away from a
generic consensus voice. Keeping them in one module makes that surface
auditable and ablatable rather than scattered across call sites.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def persona_system(
    description: str,
    values: Sequence[str] = (),
    dispositions: Sequence[str] = (),
    location: str = "New York City",
) -> str:
    """System prompt anchoring the agent to its persona (R7).

    ``values`` (what the person cares about) and ``dispositions`` (how they hold and
    voice a view) are the anchors R7 asks for: thin, label-only personas collapse
    into the model's generic assistant voice over a run, so the value/disposition
    content is stated as load-bearing, not decorative. With **no** values or
    dispositions the string is byte-identical to the P0–P4 thin prompt, so the empty
    persona survives unchanged as the R16 null-model baseline.

    ``location`` defaults to ``"New York City"`` so the P0–P5 personas are unchanged;
    the P6 census pipeline (ADR 0015 USA pivot) passes each persona's real US locale
    (e.g. a state) so the population isn't silently all-NYC.
    """
    base = (
        f"You are {description} You live in {location}. "
        "Answer the survey as this person would, in their own voice and interests."
    )
    if not values and not dispositions:
        return base
    parts = [base]
    if values:
        parts.append("These are the things you care about, and they shape how you "
                     "see this question: " + "; ".join(values) + ".")
    if dispositions:
        parts.append("This is how you engage when the topic comes up: "
                     + "; ".join(dispositions) + ".")
    parts.append("Stay in this person's voice and convictions — do not drift into a "
                 "neutral, generic tone.")
    return " ".join(parts)


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


# Prompt-level persistence clause: real people hold a considered opinion and don't
# abandon it just to match the room. Diagnosing the P6 runs showed flips are dominated
# by social-proof language ("I'm with the ladies", "the consensus here") toward the
# majority, not by new arguments — LLM sycophancy read straight off the utterances. This
# clause is the counter-pressure, opt-in (default off = byte-identical prompt) and A/B-able.
PERSISTENCE_CLAUSE = (
    "You already hold your own settled view on this, formed from your life and the kind "
    "of person you are. Hold to it. Reconsider only if you hear a genuinely new argument "
    "that actually applies to your situation — not merely because others disagree, sound "
    "confident, or seem to be the majority. Real people don't drop a considered opinion "
    "just to match the room, and neither do you.\n"
)


def action_user(
    topic: str,
    stances: Sequence[str],
    *,
    memories: Sequence[str] = (),
    mode: str = "broadcast",
    persistence: bool = False,
) -> str:
    """The per-tick action decision, prefixed with any retrieved memories.

    Abstaining is offered as a first-class choice (R25): forcing everyone to
    speak every tick is itself a homogenizing pressure, so the prompt makes
    staying quiet a legitimate option rather than a failure.

    ``mode`` (R28-adjacent discourse knob):
    - ``broadcast`` (default): agents may state a stance (SPEAK/REBUT) each turn —
      the vote-broadcasting mode whose contagion drives convergence.
    - ``deliberate``: agents only *share considerations* (reasons/stakes, no stance)
      or abstain; their position is never announced during the run and is asked
      separately at survey time. Tests whether reason-exchange, stripped of the
      per-turn vote signal, still homogenizes the eventual vote.

    ``persistence`` (default off): prepend ``PERSISTENCE_CLAUSE`` — the anti-sycophancy
    counter-pressure that tells the agent to hold its view absent a genuinely new
    argument. Off leaves the prompt byte-identical to before.
    """
    intro = f"You are talking with your neighbors about {topic}. Decide what you do this turn.\n"
    if persistence:
        intro += PERSISTENCE_CLAUSE
    if mode == "deliberate":
        body = intro + (
            '- To add to the discussion, set "action_type" to "share_consideration" and put a '
            'first-person reason or stake (one or two sentences) in "consideration" — what this '
            'means for your life. Do NOT announce which option you back; just share your '
            "perspective.\n"
            '- To stay quiet this turn, set "action_type" to "abstain" — a fine choice if you '
            "have nothing to add.\n"
        )
        return memory_block(memories) + body
    body = intro + (
        '- To speak up, set "action_type" to "speak", choose the stance that matches your '
        'view, and write a short, natural "utterance" (one or two sentences) in your own voice.\n'
        '- To share a personal consideration or stake without taking a side, set "action_type" '
        'to "share_consideration" and put a first-person reason (one or two sentences) in '
        '"consideration" — what this means for your life, not which option you back.\n'
        '- To push back against something you have heard, set "action_type" to "rebut", choose '
        'the stance you are defending, and write a short "utterance" that argues against the '
        'view you disagree with.\n'
        '- To stay quiet this turn, set "action_type" to "abstain". Say nothing if you have '
        "nothing you feel like adding — staying quiet is a fine choice.\n"
        "Available stances:\n" + "\n".join(f"- {s}" for s in stances)
    )
    return memory_block(memories) + body


# --- P6 persona-seeding pipeline (Stage 2, ADR 0016) --------------------------
#
# Two seed-time LLM steps that turn a census row + real donor-matched disposition
# items into an enactable persona. The disposition items are *fixed inputs* borrowed
# from a real survey respondent (model-free); the LLM only elaborates them, never
# invents them. The reflection sees only the measures (no demographics — withholding
# beats instructing-to-ignore); the backstory additionally gets demographics, which the
# narrative legitimately needs.

BACKSTORY_SYSTEM = (
    "You write short, grounded first-person character sketches for a social "
    "simulation. You are given fixed facts about a real-ish person and must stay "
    "strictly consistent with them — never contradict a fact, never invent a new "
    "strong trait beyond what is given."
)


def backstory_user(demographics: str, location: str, disposition_lines: Sequence[str]) -> str:
    """The backstory step: elaborate the fixed demographic + disposition inputs into
    a short first-person narrative that *enacts* the trait measures (an LLM can act a
    story more consistently than a table of scores). Demographics belong here (unlike
    the reflection step) — the narrative legitimately needs them. No survey-topic
    avoidance instruction: nothing in the inputs relates to daylight saving, so naming
    it would only prime the concept it aims to suppress."""
    traits = "\n".join(f"- {line}" for line in disposition_lines) or "- (no extra measures)"
    return (
        f"Fixed facts about this person:\n- lives in {location}\n- {demographics}\n\n"
        f"Fixed personality / disposition measures (from a matched survey respondent):\n{traits}\n\n"
        "Write a 4-5 sentence first-person backstory for this person that is fully "
        "consistent with every fact and measure above and lets the measures show "
        "through in how they live — their everyday life, in their own natural voice."
    )


REFLECTION_SYSTEM = (
    "You are a group facilitator with training in personality psychology and "
    "behavioral economics. Observing a person's measured traits, values, and "
    "attitudes, you describe how they participate in a group discussion — their "
    "temperament, how they hold and voice a view, and their risk posture. You reason "
    "only from the measures provided. You reply with strict JSON only."
)


def reflection_user(disposition_lines: Sequence[str]) -> str:
    """The reflection step (Variant A, ADR 0016): synthesize the always-on R7 anchor
    **from the real disposition measures alone**. Demographics are deliberately *not*
    shown here — the reflection's job is to translate fixed trait/attitude measures into
    behavioral implications, and it doesn't need demographics for that. Withholding them
    is a stronger safeguard than instructing the model to "ignore stereotype": the tokens
    can't bias the forward pass if they were never in the prompt. (The backstory step,
    which legitimately needs demographics, keeps them.) Returns JSON with ``values`` and
    ``dispositions`` — the shape ``persona_system`` composes."""
    traits = "\n".join(f"- {line}" for line in disposition_lines) or "- (no measures provided)"
    return (
        f"Measured traits, values, and attitudes:\n{traits}\n\n"
        "From these measures alone, describe how this person shows up in a group "
        "discussion. Reply with strict JSON of the form:\n"
        '{"values": ["..."], "dispositions": ["..."]}\n'
        "- values: 1-2 short phrases naming what this person cares about / weighs "
        "most (e.g. \"steady, concrete security over novelty\").\n"
        "- dispositions: 1-2 short phrases naming their temperament and how they hold "
        "a view — including risk posture (e.g. \"outgoing but change-averse; anchors "
        "to what has worked before\").\n"
        "JSON only, no prose."
    )


def poignancy(text: str) -> str:
    """Ask the model to rate a memory's importance 1-10 (optional path, task 4)."""
    return (
        "On a scale of 1 to 10, where 1 is purely mundane (e.g. brushing teeth) "
        "and 10 is extremely poignant (e.g. a breakup, a move), rate the likely "
        "poignancy of the following memory. Respond with a single integer only.\n\n"
        f"Memory: {text}\nRating:"
    )
