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

import random
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .actions import ACTION_SPACE_VERSION, ActionDecision, MemoryWrite, WorldUpdate
from .agent import Agent
from .feed import FeedProvider, NullFeedProvider
from .game_master import GameMaster
from .graph import run_survey
from .memory import KIND_FEED, MemoryRecord
from .questions import DST_OPTIONS
from .runlog import (
    EVENT_ACTION,
    EVENT_FEED,
    EVENT_MEMORY_WRITE,
    EVENT_RETRIEVAL,
    EVENT_TICK,
    EVENT_TICK_METRICS,
    EVENT_WORLD_UPDATE,
    RunLog,
)
from .scheduler import Scheduler, SchedulerConfig, Timing, estimate_cost_usd, run_with_retry
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
    # Information-exchange volume per tick (R12): the max number of a speaker's
    # topological neighbours that actually hear a given SPEAK. ``None`` = full reach
    # (every neighbour); a cap subsamples them so consensus pressure is tunable
    # independently of the graph's density.
    exchange_volume: int | None = None
    seed: int | None = None

    def __post_init__(self):
        if self.update_scheme not in ("simultaneous", "sequential"):
            raise ValueError(f"unknown update_scheme {self.update_scheme!r}")
        if self.exchange_volume is not None and self.exchange_volume < 0:
            raise ValueError(f"exchange_volume must be >= 0, got {self.exchange_volume}")


@dataclass
class Run:
    """A completed run's identity: the versioned config (R17) and a handle to its
    log. ``metrics`` is an explicit stub — the divergence metric (R14/R15) lands at
    P5 and will read the trajectory back out of this log. ``throughput`` is the P3
    run-level latency/token/cost aggregate (kept distinct from ``metrics`` so the
    convergence signal and the infra signal never get conflated)."""

    run_id: str
    config: dict[str, Any]
    config_hash: str
    log: RunLog
    ticks: int
    metrics: None = None  # divergence trajectory computed at P5, not P2
    throughput: dict[str, Any] | None = None  # P3 latency/token/cost aggregate

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
        # Set when built from a persona corpus (from_corpus); versioned into the run
        # config (R17) so a run's divergence is traceable to the exact persona set.
        self.corpus_meta: dict[str, Any] | None = None

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        *,
        client,
        embedder=None,
        retrieval=None,
        world: WorldState | None = None,
    ) -> "Population":
        """Build a population from a persona corpus (P6a, ADR 0016): a JSON artifact
        path, an artifact dict, or a list of ``SeededPersona``. Each persona's t=0 seed
        memories are embedded into its own private store (R2); the corpus's content
        hash + generation provenance are stashed on ``corpus_meta`` for R17.

        No seed-time LLM calls happen here — the corpus was generated once by the
        Stage-2 pipeline and cached — so building a population is free and deterministic.
        """
        from .embeddings import EmbeddingModel
        from .memory_seeds import build_store
        from .persona_pipeline import corpus_from_dict, load_corpus

        if isinstance(corpus, str):
            corpus = load_corpus(corpus)
        if isinstance(corpus, dict):
            meta = dict(corpus.get("meta", {}))
            seeded = corpus_from_dict(corpus)
        else:
            meta = {}
            seeded = list(corpus)
        embedder = embedder or EmbeddingModel()
        agents = [
            Agent(sp.persona, client, embedder=embedder,
                  memory=build_store(embedder, list(sp.memories)), retrieval=retrieval)
            for sp in seeded
        ]
        pop = cls(agents, world=world)
        pop.corpus_meta = meta
        return pop

    def survey(self, question: SurveyQuestion, *, return_skipped: bool = False):
        """Survey the live population via the LangGraph fan-out (R18/R22). With no
        ticks run this is the R16 null-model baseline. ``return_skipped=True`` also
        returns the ids of agents skipped after exhausting retries (``(answers,
        skipped)``), so a large survey's coverage is auditable rather than a silently
        short list."""
        return run_survey(self.agents, question, return_skipped=return_skipped)


@dataclass
class _CallRecord:
    """One decide call's cost, gathered for the per-tick and run-level throughput
    aggregates (P3). Token counts are the logged truth; latency is wall time."""

    agent_id: str
    latency_s: float
    attempts: int
    ok: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


def _usage_tokens(usage: dict[str, Any] | None) -> tuple[int, int, int]:
    if not usage:
        return 0, 0, 0
    return (
        int(usage.get("prompt_tokens") or 0),
        int(usage.get("completion_tokens") or 0),
        int(usage.get("total_tokens") or 0),
    )


class Simulation:
    def __init__(
        self,
        population: Population,
        *,
        topology: Topology = fully_connected,
        dynamics: DynamicsConfig | None = None,
        logger: RunLog | None = None,
        game_master: GameMaster | None = None,
        scheduler: Scheduler | None = None,
        scheduler_config: SchedulerConfig | None = None,
        feed: FeedProvider | None = None,
    ):
        self.population = population
        self.topology = topology
        self.dynamics = dynamics or DynamicsConfig()
        self.log = logger or RunLog.open(":memory:")
        self.gm = game_master or GameMaster()
        # External-signal feed (R3), off by default: NullFeedProvider injects
        # nothing, so a run is a closed system unless a feed is deliberately passed.
        self.feed = feed or NullFeedProvider()
        # The scheduler owns the concurrent decide phase (R5). Pass a Scheduler, or
        # a SchedulerConfig to tune concurrency; the config is echoed into the run
        # config (R17) so throughput is traceable to its knobs.
        self.scheduler = scheduler or Scheduler(scheduler_config)

    def run(self, ticks: int, *, on_tick=None) -> Run:
        """Run the tick loop. ``on_tick`` is an optional progress callback fired after
        each tick with ``(tick_index, cumulative_decides, elapsed_s)`` — a UI/logging
        hook (the per-decide granularity is the scheduler's ``on_progress``)."""
        config = self._build_config(ticks)
        run_id = self.log.log_run(config)
        records: list[_CallRecord] = []
        wall_start = time.perf_counter()
        for t in range(ticks):
            self._tick(run_id, t, records)
            if on_tick is not None:
                on_tick(t, len(records), time.perf_counter() - wall_start)
        wall_s = time.perf_counter() - wall_start
        return Run(
            run_id=run_id,
            config=config,
            config_hash=self.log.get_run(run_id)["config_hash"],
            log=self.log,
            ticks=ticks,
            throughput=self._aggregate_throughput(records, wall_s),
        )

    # --- one tick, dispatched on the R28 update scheme ------------------------

    def _tick(self, run_id: str, tick: int, records: list[_CallRecord]) -> None:
        self.log.log_event(run_id, event_type=EVENT_TICK, payload={"tick": tick}, tick=tick)
        # Environment step (R3): deliver any external shared signal *before* agents
        # decide, so an agent can react to the day's feed this tick. Uniform pre-tick
        # for all agents, so it is independent of the R28 within-tick scheme.
        self._deliver_feed(run_id, tick)
        decide_start = time.perf_counter()
        if self.dynamics.update_scheme == "simultaneous":
            tick_records = self._tick_simultaneous(run_id, tick)
        else:
            tick_records = self._tick_sequential(run_id, tick)
        self._log_tick_metrics(run_id, tick, tick_records, time.perf_counter() - decide_start)
        records.extend(tick_records)
        self.population.world.advance_tick()

    def _tick_simultaneous(self, run_id: str, tick: int) -> list[_CallRecord]:
        # Decide phase: every agent decides from the *same* pre-tick snapshot, each
        # reading only its own private memory (R2) — so the calls are independent
        # and fan out concurrently (R5). No memory is written yet, so no agent hears
        # another this tick. All log writes stay on this thread (serial-writer
        # invariant, ADR 0006): the scheduler returns results, we log them in order.
        snapshot = self.population.world.view()
        units = [
            (agent, self._decide_unit(agent, snapshot, tick)) for agent in self.population.agents
        ]
        outcomes = self.scheduler.map(units)
        decisions = []
        records = []
        for agent, decision, timing in outcomes:
            decision = self._require_decision(agent, tick, decision, timing)
            records.append(self._log_decision(run_id, tick, agent, decision, timing))
            decisions.append((agent, decision))
        # Resolve phase: apply all effects atomically after everyone has decided.
        for agent, decision in decisions:
            effects = self._resolve(agent, decision, tick)
            self._apply_and_log(run_id, tick, agent, effects)
        return records

    def _tick_sequential(self, run_id: str, tick: int) -> list[_CallRecord]:
        # Each agent decides against the latest state (including this tick's prior
        # applies), then its effects land immediately for the next agent to see.
        # Serial by definition (R28), so no concurrency — but still timed/retried
        # identically via run_with_retry so throughput is logged the same way.
        records = []
        for agent in self.population.agents:
            view = self.population.world.view()
            decision, timing = run_with_retry(
                self._decide_unit(agent, view, tick), self.scheduler.config
            )
            decision = self._require_decision(agent, tick, decision, timing)
            records.append(self._log_decision(run_id, tick, agent, decision, timing))
            effects = self._resolve(agent, decision, tick)
            self._apply_and_log(run_id, tick, agent, effects)
        return records

    # --- decide / resolve / apply, each logged -------------------------------

    def _decide_unit(self, agent: Agent, world_view, tick: int) -> Callable[[], ActionDecision]:
        """A zero-arg decide closure for one agent this tick — the R5 unit the
        scheduler (or the sequential path) runs. Captures nothing shared but the
        read-only world view; each call touches only its agent's private memory."""

        def decide() -> ActionDecision:
            return agent.act(
                topic=self.dynamics.topic,
                stances=self.dynamics.stances,
                world_view=world_view,
                now=float(tick),
            )

        return decide

    def _require_decision(self, agent, tick, decision, timing: Timing) -> ActionDecision:
        """A decide that exhausted its retries has no action to resolve — fail the
        run loudly (a throughput finding), not silently (which would distort
        dynamics). Retries make this rare under normal rate-limiting."""
        if decision is None or not timing.ok:
            raise RuntimeError(
                f"agent {agent.persona.id!r} decide failed at tick {tick} after "
                f"{timing.attempts} attempts: {timing.error}"
            )
        return decision

    def _log_decision(self, run_id, tick, agent, decision: ActionDecision, timing: Timing):
        action = decision.action
        prompt_t, completion_t, total_t = _usage_tokens(decision.usage)
        self.log.log_event(
            run_id, event_type=EVENT_ACTION, tick=tick, agent_id=agent.persona.id,
            payload={
                "action_type": action.action_type.value,
                "stance": action.stance,
                "utterance": action.utterance,
                "consideration": action.consideration,
                # P3 throughput fields: model pinned per call (R6), token usage, latency.
                "model": decision.model,
                "latency_s": timing.latency_s,
                "attempts": timing.attempts,
                "prompt_tokens": prompt_t,
                "completion_tokens": completion_t,
                "total_tokens": total_t,
            },
        )
        self.log.log_event(  # R29 provenance
            run_id, event_type=EVENT_RETRIEVAL, tick=tick, agent_id=agent.persona.id,
            payload=decision.provenance.to_payload(),
        )
        return _CallRecord(
            agent_id=agent.persona.id,
            latency_s=timing.latency_s,
            attempts=timing.attempts,
            ok=timing.ok,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            total_tokens=total_t,
        )

    def _resolve(self, agent, decision, tick):
        neighbors = self.topology(agent.persona.id, self.population.roster)
        neighbors = self._sample_reach(neighbors, agent.persona.id, tick)
        return self.gm.resolve(
            decision.action,
            actor_label=agent.persona.description,
            neighbors=neighbors,
            now=float(tick),
        )

    def _sample_reach(self, neighbors: list[str], agent_id: str, tick: int) -> list[str]:
        """Cap a SPEAK's reach to ``exchange_volume`` neighbours (R12). The subset is
        drawn deterministically from ``(seed, tick, agent_id)`` so a capped run stays
        reproducible and versioned. Runs in the serial resolve phase, so no lock is
        needed. ``None`` (or a cap >= degree) leaves the neighbour set untouched."""
        k = self.dynamics.exchange_volume
        if k is None or len(neighbors) <= k:
            return neighbors
        rng = random.Random(f"{self.dynamics.seed}-{tick}-{agent_id}")
        return rng.sample(neighbors, k)

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

    def _deliver_feed(self, run_id: str, tick: int) -> None:
        """Inject external shared signals (e.g. X posts) into targeted agents (R3).

        An *environment → agent* path, distinct from the agent → agent SPEAK the
        Game Master resolves: the feed lands as a ``KIND_FEED`` memory in the target
        agent's own store and is logged as its own ``feed_delivery`` stream so its
        causal effect on convergence stays traceable. Unknown targets are skipped.
        """
        events = self.feed.events_for_tick(tick, self.population.roster, self.population.world.view())
        for event in events:
            target = self.population.by_id.get(event.target_agent_id)
            if target is None:
                continue  # feed aimed at an agent not in this run — drop, don't crash
            now = float(tick) if event.created_at is None else event.created_at
            target.memory.add(
                MemoryRecord(
                    text=event.text,
                    embedding=target.embedder.encode(event.text),
                    importance=event.importance,
                    created_at=now,
                    last_accessed_at=now,
                    kind=KIND_FEED,
                )
            )
            self.log.log_event(
                run_id, event_type=EVENT_FEED, tick=tick,
                agent_id=event.target_agent_id, payload=event.to_payload(),
            )

    # --- throughput accounting (P3) ------------------------------------------

    def _log_tick_metrics(
        self, run_id: str, tick: int, records: list[_CallRecord], decide_wall_s: float
    ) -> None:
        """Per-tick throughput summary — the trajectory view (R15-style): latency
        and token cost logged each tick, not just at endpoint."""
        latencies = [r.latency_s for r in records]
        self.log.log_event(
            run_id, event_type=EVENT_TICK_METRICS, tick=tick,
            payload={
                "tick": tick,
                "n_calls": len(records),
                "decide_wall_s": decide_wall_s,
                "prompt_tokens": sum(r.prompt_tokens for r in records),
                "completion_tokens": sum(r.completion_tokens for r in records),
                "total_tokens": sum(r.total_tokens for r in records),
                "latency_mean_s": _mean(latencies),
                "latency_max_s": max(latencies, default=0.0),
                "retries": sum(r.attempts - 1 for r in records),
                "failures": sum(not r.ok for r in records),
                "update_scheme": self.dynamics.update_scheme,
                "max_concurrency": self.scheduler.config.max_concurrency,
            },
        )

    def _aggregate_throughput(self, records: list[_CallRecord], wall_s: float) -> dict[str, Any]:
        """Run-level throughput aggregate (P3), returned on ``Run.throughput``. The
        headline is ``decides_per_s`` = calls / wall — the answer to the phase-3
        spike ('what agent count / tick rate is sustainable')."""
        cfg = self.scheduler.config
        latencies = sorted(r.latency_s for r in records)
        prompt_t = sum(r.prompt_tokens for r in records)
        completion_t = sum(r.completion_tokens for r in records)
        n_calls = len(records)
        return {
            "n_agents": len(self.population.agents),
            "n_calls": n_calls,
            "wall_s": wall_s,
            "decides_per_s": (n_calls / wall_s) if wall_s > 0 else None,
            "prompt_tokens": prompt_t,
            "completion_tokens": completion_t,
            "total_tokens": sum(r.total_tokens for r in records),
            "latency_mean_s": _mean(latencies),
            "latency_p95_s": _percentile(latencies, 0.95),
            "latency_max_s": max(latencies, default=0.0),
            "retries": sum(r.attempts - 1 for r in records),
            "failures": sum(not r.ok for r in records),
            "est_cost_usd": estimate_cost_usd(cfg.price_per_mtok, prompt_t, completion_t),
            "max_concurrency": cfg.max_concurrency,
            "executor": cfg.executor,
        }

    def _topology_config(self) -> dict[str, Any] | str:
        """A structured descriptor for the topology if it exposes one (the P4
        ``Topology`` classes do), else the legacy name of a plain callable seam."""
        to_config = getattr(self.topology, "to_config", None)
        if callable(to_config):
            return to_config()
        return getattr(self.topology, "__name__", "custom")

    def _committed_config(self) -> list[dict[str, str]]:
        """The committed-minority roster (R11), sorted by id — the fixed faction
        whose stance is immovable, recorded so its effect on persistence is traceable."""
        return sorted(
            (
                {"id": a.persona.id, "stance": a.committed_stance}
                for a in self.population.agents
                if getattr(a, "committed_stance", None) is not None
            ),
            key=lambda d: d["id"],
        )

    def _build_config(self, ticks: int) -> dict[str, Any]:
        """The versioned run config (R17): everything that could change the run's
        trajectory, so an observed convergence is traceable to its cause."""
        agents = self.population.agents
        client_cfg = agents[0].client.config
        cfg = self.scheduler.config
        return {
            "ticks": ticks,
            "action_space_version": ACTION_SPACE_VERSION,
            "update_scheme": self.dynamics.update_scheme,
            "topic": self.dynamics.topic,
            "stances": list(self.dynamics.stances),
            # Topology is a versioned run parameter (R4/R17): a structured descriptor
            # (name + graph params + seed) when the topology exposes one, so an observed
            # convergence is traceable to the exact graph, not just its name.
            "topology": self._topology_config(),
            # Consensus-pressure knobs (R12/R11), versioned so their effect is traceable.
            "exchange_volume": self.dynamics.exchange_volume,
            "committed": self._committed_config(),
            # The external-signal source is part of the versioned config (R3/R17):
            # an observed convergence must be attributable to whether a feed ran.
            "feed_provider": type(self.feed).__name__,
            "seed": self.dynamics.seed,
            "model": client_cfg.model,
            # Provider pinned & logged per run (R6) — model id alone isn't enough
            # once the backend switches (OpenRouter now → self-hosted vLLM at P5).
            "provider": {
                "base_url": getattr(client_cfg, "base_url", None),
                "model": client_cfg.model,
            },
            # Throughput knobs echoed so an observed latency/cost is traceable (R17).
            "scheduler": {
                "max_concurrency": cfg.max_concurrency,
                "max_retries": cfg.max_retries,
                "executor": cfg.executor,
            },
            "retrieval": asdict(agents[0].retrieval),
            # When the population came from a seeded corpus (P6a), cite its version +
            # content hash (R17) so the run is traceable to the exact persona artifact.
            "persona_corpus": getattr(self.population, "corpus_meta", None),
            # Persona content is versioned (R17): a thick-vs-thin run's divergence must
            # be traceable to the value/disposition anchors (R7), not just the id.
            "personas": [
                {
                    "id": a.persona.id,
                    "description": a.persona.description,
                    "temperature": a.persona.temperature,
                    "values": list(a.persona.values),
                    "dispositions": list(a.persona.dispositions),
                }
                for a in agents
            ],
        }


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(sorted_xs: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an already-sorted sequence (0.0 if empty)."""
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, int(round(q * (len(sorted_xs) - 1))))
    return sorted_xs[idx]
