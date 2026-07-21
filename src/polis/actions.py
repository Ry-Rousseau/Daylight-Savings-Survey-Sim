"""The action space — closed, versioned, structured (R23), plus the effect and
provenance types the Game Master and run log speak in.

R23: the action space is a *closed enumeration* with structured payloads, not
open-ended free text, so it stays schema-constrainable (R20) and quantitatively
loggable (R14–R17). P2 ships exactly two action types — ``SPEAK`` and
``ABSTAIN`` — one content-exchange action plus the always-valid no-op (R25).
The enum is versioned (``ACTION_SPACE_VERSION``) so a later expansion (e.g. the
topology-mutating tie actions of R26 at P4) is a visible, logged schema change.

R29: ``RetrievalProvenance`` carries the *scored memory set* that actually
conditioned a decision — the mechanistic "why", distinct from the model's
post-hoc self-reported rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel

ACTION_SPACE_VERSION = 3


class ActionType(str, Enum):
    SPEAK = "speak"
    ABSTAIN = "abstain"
    SHARE_CONSIDERATION = "share_consideration"
    REBUT = "rebut"


class Action(BaseModel):
    """One agent's chosen action for a tick. ``stance``/``utterance`` are set for
    SPEAK and REBUT, ``consideration`` only for SHARE_CONSIDERATION; all are optional
    in the schema (OpenRouter soft-enforces) and the Game Master validates them — a
    malformed SPEAK / REBUT / SHARE_CONSIDERATION resolves as a no-op."""

    action_type: ActionType
    stance: str | None = None
    utterance: str | None = None
    consideration: str | None = None

    @classmethod
    def abstain(cls) -> "Action":
        return cls(action_type=ActionType.ABSTAIN)

    @classmethod
    def speak(cls, stance: str, utterance: str) -> "Action":
        return cls(action_type=ActionType.SPEAK, stance=stance, utterance=utterance)

    @classmethod
    def consider(cls, consideration: str) -> "Action":
        return cls(action_type=ActionType.SHARE_CONSIDERATION, consideration=consideration)

    @classmethod
    def rebut(cls, stance: str, utterance: str) -> "Action":
        return cls(action_type=ActionType.REBUT, stance=stance, utterance=utterance)

    def is_valid_speak(self) -> bool:
        return (
            self.action_type is ActionType.SPEAK
            and bool(self.stance)
            and bool(self.utterance)
        )

    def is_valid_consideration(self) -> bool:
        # A consideration carries a reason/stake but no stance; only its text must
        # be present for it to resolve (mirrors is_valid_speak's guard).
        return (
            self.action_type is ActionType.SHARE_CONSIDERATION
            and bool(self.consideration)
        )

    def is_valid_rebut(self) -> bool:
        # A rebut states a position (stance) framed as active pushback against what
        # was heard; like a SPEAK it needs both a stance and the counter-argument text.
        return (
            self.action_type is ActionType.REBUT
            and bool(self.stance)
            and bool(self.utterance)
        )


def action_json_schema(stances: list[str]) -> dict[str, Any]:
    """json_schema for constrained action decoding (R20/R23).

    ``stances`` is the closed set of DST positions a SPEAK may express, so the
    stance vocabulary is not hardcoded. ``action_type`` is required; stance,
    utterance, and consideration are optional here and enforced by the Game Master.
    """
    return {
        "type": "object",
        "properties": {
            "action_type": {"type": "string", "enum": [t.value for t in ActionType]},
            "stance": {"type": "string", "enum": list(stances)},
            "utterance": {"type": "string"},
            "consideration": {"type": "string"},
        },
        "required": ["action_type"],
        "additionalProperties": False,
    }


# --- Effects: what a resolved action does to the world / other agents ---------
# The Game Master returns these (pure, embedding-free); the Simulation applies
# them (embedding a MemoryWrite, incrementing the tally) and logs each one.


@dataclass(frozen=True)
class MemoryWrite:
    """Append a memory to *another* agent's private store (a resolved SPEAK).
    Cross-agent memory only ever originates here, never from an agent directly."""

    target_agent_id: str
    text: str
    kind: str
    importance: float
    created_at: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "effect": "memory_write",
            "target_agent_id": self.target_agent_id,
            "text": self.text,
            "kind": self.kind,
            "importance": self.importance,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class WorldUpdate:
    """Increment the shared stance tally (a deliberately shared signal, R3)."""

    stance: str

    def to_payload(self) -> dict[str, Any]:
        return {"effect": "world_update", "stance": self.stance}


Effect = MemoryWrite | WorldUpdate


# --- Provenance: the scored memory set behind a decision (R29) ----------------


@dataclass(frozen=True)
class ProvenanceEntry:
    text: str
    kind: str
    created_at: float
    recency: float
    importance: float
    relevance: float
    total: float


@dataclass(frozen=True)
class RetrievalProvenance:
    """The mechanistic 'why' behind one decision: which memories surfaced, with
    their recency/importance/relevance components and combined score (R29)."""

    query: str
    hits: list[ProvenanceEntry]

    def to_payload(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "hits": [
                {
                    "text": h.text,
                    "kind": h.kind,
                    "created_at": h.created_at,
                    "recency": h.recency,
                    "importance": h.importance,
                    "relevance": h.relevance,
                    "total": h.total,
                }
                for h in self.hits
            ],
        }


@dataclass(frozen=True)
class ActionDecision:
    """What ``Agent.act`` returns: the chosen action plus its retrieval provenance.

    ``usage``/``model`` carry the decode's token counts and the pinned model id
    (R6) up to the scheduler for throughput/cost logging (P3). Both default to
    ``None`` so fakes and non-LLM constructions stay valid."""

    action: Action
    provenance: RetrievalProvenance
    usage: dict[str, Any] | None = None
    model: str | None = None
