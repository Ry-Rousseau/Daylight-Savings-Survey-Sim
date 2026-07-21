"""The Game Master — deterministic, non-LLM action resolution (R24).

Agents *propose* structured actions (R23); the Game Master *resolves* them into
concrete effects with no further model call. Every action type shipped has an
explicit branch here (R24) — nothing is added to the action space speculatively
without resolution logic. Resolution is pure: it returns effects and mutates
nothing, so it is fully testable without a world, a memory store, or a network.

The R2 boundary lives here: a SPEAK becomes a ``MemoryWrite`` into the *listener's*
store, never the speaker's — cross-agent memory only ever originates from a
GM-resolved effect.
"""
from __future__ import annotations

from collections.abc import Sequence

from .actions import Action, ActionType, Effect, MemoryWrite, WorldUpdate
from .memory import KIND_HEARD

# Importance assigned to a "heard from another agent" memory. Mid-scale (Park
# 1–10) — hearsay is neither trivial nor self-authored; exposed here rather than
# buried as a literal so it can be tuned when dynamics work begins (P4).
HEARD_IMPORTANCE = 5.0


class GameMaster:
    def resolve(
        self,
        action: Action,
        *,
        actor_label: str,
        neighbors: Sequence[str],
        now: float,
    ) -> list[Effect]:
        """Resolve one agent's action into effects (R24).

        ``actor_label`` is the speaker's persona description (what listeners
        remember hearing from); ``neighbors`` are the listener agent ids the
        SPEAK reaches under the current topology.
        """
        if action.action_type is ActionType.ABSTAIN:
            return self._resolve_abstain()
        if action.action_type is ActionType.SPEAK:
            return self._resolve_speak(action, actor_label, neighbors, now)
        if action.action_type is ActionType.SHARE_CONSIDERATION:
            return self._resolve_consideration(action, actor_label, neighbors, now)
        raise ValueError(f"No resolution logic for action type {action.action_type!r} (R24)")

    def _resolve_abstain(self) -> list[Effect]:
        return []  # R25: no-op is always valid; forcing action would homogenize

    def _resolve_speak(
        self, action: Action, actor_label: str, neighbors: Sequence[str], now: float
    ) -> list[Effect]:
        # A malformed SPEAK (missing stance or utterance) degrades to a no-op
        # rather than corrupting the stream.
        if not action.is_valid_speak():
            return []
        effects: list[Effect] = [
            MemoryWrite(
                target_agent_id=listener,
                text=f"{actor_label} said: {action.utterance}",
                kind=KIND_HEARD,
                importance=HEARD_IMPORTANCE,
                created_at=now,
            )
            for listener in neighbors
        ]
        effects.append(WorldUpdate(stance=action.stance))
        return effects

    def _resolve_consideration(
        self, action: Action, actor_label: str, neighbors: Sequence[str], now: float
    ) -> list[Effect]:
        # A consideration circulates a persona-specific reason/stake, not a vote:
        # it reaches listeners exactly as a SPEAK does, but emits *no* WorldUpdate,
        # so it never touches the shared stance tally (that is the point — reasons,
        # not votes, so hearing them cannot drive stance conformity). A malformed
        # consideration (no text) degrades to a no-op, like a malformed SPEAK.
        if not action.is_valid_consideration():
            return []
        return [
            MemoryWrite(
                target_agent_id=listener,
                text=f"{actor_label} shared: {action.consideration}",
                kind=KIND_HEARD,
                importance=HEARD_IMPORTANCE,
                created_at=now,
            )
            for listener in neighbors
        ]
