"""Interaction topology — pluggable, seeded, swappable graphs (R4/R10/R13).

Layer-3 Dynamics. The topology decides, for a *speaking* agent, which other
agents hear it — the graph the SPEAK contagion flows over. It is a **pluggable
parameter** (R4), never hardcoded into the tick loop: ``Simulation`` holds one
``Topology`` and asks it for each speaker's listeners. Swapping the object (same
personas, new graph) is the R13 counterfactual; the adjacency is built once and
**frozen**, so the graph does not drift within a run.

Each concrete topology is **deterministic given its** ``seed``, so a run is
reproducible and the graph is versioned into the run config (R17 via
:meth:`Topology.to_config`). Graphs are **undirected**: if A hears B then B
hears A — an edge is a mutual channel, matching SPEAK semantics (one SPEAK
becomes a MemoryWrite to each listener).

Defaulting to sparse/structured graphs over full connectivity is R10 — full
exposure biases toward global consensus. ``StochasticBlock`` (clustered
communities) and ``SmallWorld`` exist precisely to preserve disparate
communities; ``FullyConnected`` is kept as the high-consensus-pressure control.

A ``Topology`` is a drop-in for the existing callable seam
(``Callable[[str, tuple[str, ...]], list[str]]`` in ``simulation.py``): instances
are callable, so ``sim.topology = SmallWorld(...)`` just works.
"""
from __future__ import annotations

import random
from itertools import combinations
from typing import Any

Roster = tuple[str, ...]


def _symmetric_adjacency(
    roster: Roster, edges
) -> dict[str, tuple[str, ...]]:
    """Fold an iterable of unordered ``(a, b)`` id pairs into a frozen, symmetric
    adjacency map. Self-loops are dropped; each neighbour list is sorted for a
    stable, reproducible order."""
    adj: dict[str, set[str]] = {a: set() for a in roster}
    for a, b in edges:
        if a == b:
            continue
        adj[a].add(b)
        adj[b].add(a)
    return {a: tuple(sorted(neigh)) for a, neigh in adj.items()}


class Topology:
    """Base class: a callable ``(agent_id, roster) -> list[str]`` returning the ids
    that hear ``agent_id`` speak.

    Adjacency is computed once for a given roster and cached (frozen for the run,
    R13). Subclasses implement :meth:`_build_edges`; the base handles caching,
    symmetry, and the callable/config protocol.
    """

    name = "topology"

    def __init__(self, *, seed: int | None = None):
        self.seed = seed
        self._adj: dict[str, tuple[str, ...]] | None = None
        self._roster: Roster | None = None

    def _ensure(self, roster: Roster) -> dict[str, tuple[str, ...]]:
        # Rebuild only if this is a new/different roster; otherwise the graph is
        # frozen (R13) — repeated calls within a run return the same adjacency.
        if self._adj is None or self._roster != roster:
            self._adj = _symmetric_adjacency(roster, self._build_edges(roster))
            self._roster = roster
        return self._adj

    def neighbors(self, agent_id: str, roster: Roster) -> list[str]:
        return list(self._ensure(roster).get(agent_id, ()))

    def __call__(self, agent_id: str, roster: Roster) -> list[str]:
        return self.neighbors(agent_id, roster)

    def _build_edges(self, roster: Roster):
        raise NotImplementedError

    def to_config(self) -> dict[str, Any]:
        """The graph's identity for the versioned run config (R17)."""
        return {"name": self.name, "seed": self.seed}


class FullyConnected(Topology):
    """Everyone hears everyone — the high-consensus-pressure control (R10 warns
    against defaulting to it). Behaviour-equal to the free ``fully_connected`` in
    ``simulation.py``; provided as a class so it logs a structured config."""

    name = "fully_connected"

    def _build_edges(self, roster: Roster):
        return combinations(roster, 2)

    def to_config(self) -> dict[str, Any]:
        return {"name": self.name}


class RingLattice(Topology):
    """A regular ring: each node tied to its ``k`` nearest neighbours on a cycle
    (``k//2`` on each side). Sparse and highly clustered with a long characteristic
    path — a slow-mixing graph. Deterministic; ``seed`` is unused (the ring is
    fixed by roster order) but accepted for a uniform constructor."""

    name = "ring_lattice"

    def __init__(self, k: int = 2, *, seed: int | None = None):
        super().__init__(seed=seed)
        if k % 2 != 0:
            raise ValueError(f"ring lattice degree k must be even, got {k}")
        if k < 2:
            raise ValueError(f"ring lattice degree k must be >= 2, got {k}")
        self.k = k

    def _build_edges(self, roster: Roster):
        n = len(roster)
        half = min(self.k // 2, (n - 1) // 2)  # clamp for small rosters
        return [
            (roster[i], roster[(i + j) % n])
            for i in range(n)
            for j in range(1, half + 1)
        ]

    def to_config(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k}


class SmallWorld(Topology):
    """Watts–Strogatz small-world: a ``k``-regular ring whose edges are each
    rewired to a random target with probability ``p``. Low ``p`` keeps the ring's
    high clustering while the few long-range rewires collapse the path length —
    the classic 'sparse but short-path' regime that spreads a signal fast without
    full connectivity. Deterministic given ``seed``."""

    name = "small_world"

    def __init__(self, k: int = 4, p: float = 0.1, *, seed: int | None = None):
        super().__init__(seed=seed)
        if k % 2 != 0:
            raise ValueError(f"small-world degree k must be even, got {k}")
        if k < 2:
            raise ValueError(f"small-world degree k must be >= 2, got {k}")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"rewire probability p must be in [0, 1], got {p}")
        self.k = k
        self.p = p

    def _build_edges(self, roster: Roster):
        n = len(roster)
        half = min(self.k // 2, (n - 1) // 2)
        rng = random.Random(self.seed)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(1, half + 1):
                nb = (i + j) % n
                adj[i].add(nb)
                adj[nb].add(i)
        # Rewire, ring-distance by ring-distance (the WS ordering): for each edge
        # (i, i+j), with prob p move its far endpoint to a random non-neighbour.
        for j in range(1, half + 1):
            for i in range(n):
                old = (i + j) % n
                if old not in adj[i]:
                    continue  # already rewired away
                if rng.random() >= self.p:
                    continue
                forbidden = adj[i] | {i}
                candidates = [x for x in range(n) if x not in forbidden]
                if not candidates:
                    continue
                new = rng.choice(candidates)
                adj[i].discard(old)
                adj[old].discard(i)
                adj[i].add(new)
                adj[new].add(i)
        return [
            (roster[a], roster[b]) for a in range(n) for b in adj[a] if a < b
        ]

    def to_config(self) -> dict[str, Any]:
        return {"name": self.name, "k": self.k, "p": self.p, "seed": self.seed}


class StochasticBlock(Topology):
    """Stochastic block model — clustered communities. The roster is split into
    ``n_blocks`` contiguous blocks; a within-block pair is tied with probability
    ``p_in``, a cross-block pair with ``p_out`` (``p_in`` >> ``p_out`` → dense
    communities loosely bridged). The graph for preserving disparate communities
    under interaction (R10) and for seeding committed factions per block (R11).
    Deterministic given ``seed``."""

    name = "stochastic_block"

    def __init__(
        self,
        n_blocks: int = 2,
        p_in: float = 0.8,
        p_out: float = 0.02,
        *,
        seed: int | None = None,
    ):
        super().__init__(seed=seed)
        if n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {n_blocks}")
        for name, p in (("p_in", p_in), ("p_out", p_out)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {p}")
        self.n_blocks = n_blocks
        self.p_in = p_in
        self.p_out = p_out

    def block_assignments(self, roster: Roster) -> dict[str, int]:
        """Map each agent id to its block index (contiguous split). Exposed so a
        run can seed a committed minority *within* one block (R11)."""
        n = len(roster)
        return {roster[i]: (i * self.n_blocks) // n for i in range(n)}

    def _build_edges(self, roster: Roster):
        n = len(roster)
        rng = random.Random(self.seed)
        block = [(i * self.n_blocks) // n for i in range(n)]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                p = self.p_in if block[i] == block[j] else self.p_out
                if rng.random() < p:
                    edges.append((roster[i], roster[j]))
        return edges

    def to_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_blocks": self.n_blocks,
            "p_in": self.p_in,
            "p_out": self.p_out,
            "seed": self.seed,
        }
