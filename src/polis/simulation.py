"""The Simulation / Population container and the custom tick loop (R22, R28).

``Population`` ties the three state tiers together: it owns the agents (tier-1
private memory) and the shared ``WorldState`` (tier 2). ``Simulation`` hosts the
**custom tick loop** — the non-LangGraph half of R22; LangGraph stays scoped to
the bounded survey fan-out (``Population.survey``) and does not run the sim core.

Within a tick, the read/write ordering is an explicit run parameter (R28):

- ``simultaneous`` (default): every agent decides from the *same* pre-tick state;
  all Game-Master effects are applied atomically after the decide phase, so no
  agent hears another agent's utterance within the same tick. Lower within-tick
  contagion — a cleaner convergence baseline.
- ``sequential``: each agent decides, its effects are applied immediately, and the
  next agent decides already seeing them. Higher within-tick contagion.

The choice changes convergence, so it is recorded in the run config (R17), never
left to implementation accident (R28). Every tick's actions, effects, and per-
decision retrieval provenance (R29) are written to the durable run log (tier 3).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import ACTION_SPACE_VERSION, MemoryWrite, WorldUpdate
from .agent import Agent
from .game_master import GameMaster
from .graph import run_survey
from .memory import MemoryRecord
from .questions import DST_OPTIONS
from .runlog import (
    EVENT_ACTION,
    EVENT_MEMORY_WRITE,
    EVENT_RETRIEVAL,
    EVENT_TICK,
    EVENT_WORLD_UPDATE,
    RunLog,
)
from .survey import SurveyQuestion
from .world import WorldState

Topology = Callable[[str, tuple[str, ...]], list[str]]


def fully_connected(agent_id: str, roster: tuple[str, ...]) -> list[str]:
    """Trivial P2 topology: everyone hears everyone. The swap point for the
    pluggable topologies of R4/R10–R13 at P4."""
    return [a for a in roster if a != agent_id]


@dataclass
class DynamicsConfig:
    """Tunable dynamics parameters, recorded in the run config (R17/R28)."""

    update_scheme: str = "simultaneous"  # or "sequential" (R28)
    topic: str = "daylight saving time"
    stances: Sequence[str] = field(default_factory=lambda: list(DST_OPTIONS))
    seed: int | None = None

    def __post_init__(self):
        if self.update_scheme not in ("simultaneous", "sequential"):
            raise ValueError(f"unknown update_scheme {self.update_scheme!r}")


@dataclass
class Run:
    """A completed run's identity: the versioned config (R17) and a handle to its
    log. ``metrics`` is an explicit stub in P2 — the divergence metric (R14/R15)
    lands at P5 and will read the trajectory back out of this log."""

    run_id: str
    config: dict[str, Any]
    config_hash: str
    log: RunLog
    ticks: int
    metrics: None = None  # divergence trajectory computed at P5, not P2

    def events(self, **kw):
        return self.log.events(self.run_id, **kw)


class Population:
    """Owns the agents (tier 1) and the shared world state (tier 2)."""

    def __init__(self, agents: Sequence[Agent], *, world: WorldState | None = None):
        if not agents:
            raise ValueError("Population needs at least one agent")
        self.agents = list(agents)
        self.by_id = {a.persona.id: a for a in self.agents}
        self.roster = tuple(self.by_id)
        self.world = world or WorldState(roster=self.roster)

    def survey(self, question: SurveyQuestion) -> list[dict]:
        """Survey the live population via the LangGraph fan-out (R18/R22). With no
        ticks run this is the R16 null-model baseline."""
        return run_survey(self.agents, question)


class Simulation:
    def __init__(
        self,
        population: Population,
        *,
        topology: Topology = fully_connected,
        dynamics: DynamicsConfig | None = None,
        logger: RunLog | None = None,
        game_master: GameMaster | None = None,
    ):
        self.population = population
        self.topology = topology
        self.dynamics = dynamics or DynamicsConfig()
        self.log = logger or RunLog.open(":memory:")
        self.gm = game_master or GameMaster()

    def run(self, ticks: int) -> Run:
        config = self._build_config(ticks)
        run_id = self.log.log_run(config)
        for t in range(ticks):
            self._tick(run_id, t)
        return Run(
            run_id=run_id,
            config=config,
            config_hash=self.log.get_run(run_id)["config_hash"],
            log=self.log,
            ticks=ticks,
        )

    # --- one tick, dispatched on the R28 update scheme ------------------------

    def _tick(self, run_id: str, tick: int) -> None:
        self.log.log_event(run_id, event_type=EVENT_TICK, payload={"tick": tick}, tick=tick)
        if self.dynamics.update_scheme == "simultaneous":
            self._tick_simultaneous(run_id, tick)
        else:
            self._tick_sequential(run_id, tick)
        self.population.world.advance_tick()

    def _tick_simultaneous(self, run_id: str, tick: int) -> None:
        # Decide phase: every agent decides from the pre-tick snapshot; no memory
        # is written yet, so no agent hears another this tick.
        snapshot = self.population.world.view()
        decisions = []
        for agent in self.population.agents:
            decision = self._decide_and_log(run_id, tick, agent, snapshot)
            decisions.append((agent, decision))
        # Resolve phase: apply all effects atomically after everyone has decided.
        for agent, decision in decisions:
            effects = self._resolve(agent, decision, tick)
            self._apply_and_log(run_id, tick, agent, effects)

    def _tick_sequential(self, run_id: str, tick: int) -> None:
        # Each agent decides against the latest state (including this tick's prior
        # applies), then its effects land immediately for the next agent to see.
        for agent in self.population.agents:
            view = self.population.world.view()
            decision = self._decide_and_log(run_id, tick, agent, view)
            effects = self._resolve(agent, decision, tick)
            self._apply_and_log(run_id, tick, agent, effects)

    # --- decide / resolve / apply, each logged -------------------------------

    def _decide_and_log(self, run_id, tick, agent, world_view):
        decision = agent.act(
            topic=self.dynamics.topic,
            stances=self.dynamics.stances,
            world_view=world_view,
            now=float(tick),
        )
        action = decision.action
        self.log.log_event(
            run_id, event_type=EVENT_ACTION, tick=tick, agent_id=agent.persona.id,
            payload={
                "action_type": action.action_type.value,
                "stance": action.stance,
                "utterance": action.utterance,
            },
        )
        self.log.log_event(  # R29 provenance
            run_id, event_type=EVENT_RETRIEVAL, tick=tick, agent_id=agent.persona.id,
            payload=decision.provenance.to_payload(),
        )
        return decision

    def _resolve(self, agent, decision, tick):
        neighbors = self.topology(agent.persona.id, self.population.roster)
        return self.gm.resolve(
            decision.action,
            actor_label=agent.persona.description,
            neighbors=neighbors,
            now=float(tick),
        )

    def _apply_and_log(self, run_id, tick, actor, effects) -> None:
        for effect in effects:
            if isinstance(effect, MemoryWrite):
                self._apply_memory_write(effect)
                event_type = EVENT_MEMORY_WRITE
            elif isinstance(effect, WorldUpdate):
                self.population.world.record_stance(effect.stance)
                event_type = EVENT_WORLD_UPDATE
            else:  # pragma: no cover - closed effect set
                raise TypeError(f"unknown effect {effect!r}")
            self.log.log_event(
                run_id, event_type=event_type, tick=tick,
                agent_id=actor.persona.id, payload=effect.to_payload(),
            )

    def _apply_memory_write(self, effect: MemoryWrite) -> None:
        """Deliver a heard memory into the *target* agent's private store. This is
        the only path by which one agent's output enters another's memory (R2)."""
        target = self.population.by_id[effect.target_agent_id]
        target.memory.add(
            MemoryRecord(
                text=effect.text,
                embedding=target.embedder.encode(effect.text),
                importance=effect.importance,
                created_at=effect.created_at,
                last_accessed_at=effect.created_at,
                kind=effect.kind,
            )
        )

    def _build_config(self, ticks: int) -> dict[str, Any]:
        """The versioned run config (R17): everything that could change the run's
        trajectory, so an observed convergence is traceable to its cause."""
        agents = self.population.agents
        return {
            "ticks": ticks,
            "action_space_version": ACTION_SPACE_VERSION,
            "update_scheme": self.dynamics.update_scheme,
            "topic": self.dynamics.topic,
            "stances": list(self.dynamics.stances),
            "topology": getattr(self.topology, "__name__", "custom"),
            "seed": self.dynamics.seed,
            "model": agents[0].client.config.model,
            "retrieval": asdict(agents[0].retrieval),
            "personas": [
                {
                    "id": a.persona.id,
                    "description": a.persona.description,
                    "temperature": a.persona.temperature,
                }
                for a in agents
            ],
        }
