"""Topology tests — pluggable, seeded, symmetric, frozen graphs (R4/R10/R13).

Pure graph structure: no agents, no network. Each topology is exercised as the
callable seam ``(agent_id, roster) -> listeners`` the Simulation uses.
"""
import pytest

from polis.simulation import fully_connected
from polis.topology import (
    FullyConnected,
    RingLattice,
    SmallWorld,
    StochasticBlock,
)

ROSTER = tuple(f"a{i:02d}" for i in range(20))


def adjacency(topo, roster=ROSTER) -> dict[str, set[str]]:
    return {a: set(topo(a, roster)) for a in roster}


def edge_count(topo, roster=ROSTER) -> int:
    return sum(len(ns) for ns in adjacency(topo, roster).values()) // 2


ALL = [
    FullyConnected(),
    RingLattice(k=4),
    SmallWorld(k=4, p=0.2, seed=7),
    StochasticBlock(n_blocks=2, p_in=0.8, p_out=0.02, seed=7),
]


@pytest.mark.parametrize("topo", ALL, ids=lambda t: t.name)
def test_undirected_symmetry(topo):
    """If A hears B then B hears A — edges are mutual channels."""
    adj = adjacency(topo)
    for a, neigh in adj.items():
        assert a not in neigh  # no self-loops
        for b in neigh:
            assert a in adj[b], f"{a}->{b} not mirrored"


def test_fully_connected_matches_free_function():
    """The class is behaviour-equal to the legacy free callable."""
    for a in ROSTER:
        assert set(FullyConnected()(a, ROSTER)) == set(fully_connected(a, ROSTER))
        assert len(FullyConnected()(a, ROSTER)) == len(ROSTER) - 1


def test_ring_lattice_is_k_regular():
    topo = RingLattice(k=4)
    adj = adjacency(topo)
    assert all(len(ns) == 4 for ns in adj.values())  # exact degree k on a full ring


@pytest.mark.parametrize(
    "topo",
    [SmallWorld(k=4, p=0.3, seed=42), StochasticBlock(n_blocks=3, p_in=0.7, p_out=0.05, seed=42)],
    ids=lambda t: t.name,
)
def test_seeded_graph_is_reproducible(topo):
    """Same seed → identical graph, rebuilt from the logged config params."""
    same = type(topo)(**_params(topo))
    assert adjacency(topo) == adjacency(same)


def test_different_seed_gives_different_small_world():
    a = SmallWorld(k=4, p=0.5, seed=1)
    b = SmallWorld(k=4, p=0.5, seed=2)
    assert adjacency(a) != adjacency(b)


def test_sparsity_ordering():
    """R10: the structured graphs are strictly sparser than full connectivity."""
    full = edge_count(FullyConnected())
    assert edge_count(RingLattice(k=4)) < full
    assert edge_count(SmallWorld(k=4, p=0.2, seed=3)) < full


def test_stochastic_block_denser_within_blocks():
    """Intra-block ties are far likelier than cross-block ones (clustered communities)."""
    topo = StochasticBlock(n_blocks=2, p_in=0.9, p_out=0.03, seed=11)
    blocks = topo.block_assignments(ROSTER)
    adj = adjacency(topo)
    intra = inter = 0
    for a, neigh in adj.items():
        for b in neigh:
            if a < b:
                if blocks[a] == blocks[b]:
                    intra += 1
                else:
                    inter += 1
    # Possible pairs per class for normalisation.
    same = sum(1 for i, a in enumerate(ROSTER) for b in ROSTER[i + 1:] if blocks[a] == blocks[b])
    diff = sum(1 for i, a in enumerate(ROSTER) for b in ROSTER[i + 1:] if blocks[a] != blocks[b])
    assert intra / same > inter / diff


def test_graph_is_frozen_after_build():
    """Adjacency is cached (R13): repeated calls are identical and a returned list
    cannot be used to mutate internal state."""
    topo = SmallWorld(k=4, p=0.3, seed=5)
    first = topo("a00", ROSTER)
    first.append("tamper")
    assert "tamper" not in topo("a00", ROSTER)  # internal state untouched
    assert topo("a00", ROSTER) == sorted(topo("a00", ROSTER))  # stable order


def test_to_config_carries_params():
    assert SmallWorld(k=6, p=0.1, seed=9).to_config() == {
        "name": "small_world", "k": 6, "p": 0.1, "seed": 9}
    assert FullyConnected().to_config() == {"name": "fully_connected"}


@pytest.mark.parametrize(
    "ctor",
    [
        lambda: RingLattice(k=3),
        lambda: SmallWorld(k=3),
        lambda: SmallWorld(k=4, p=1.5),
        lambda: StochasticBlock(n_blocks=0),
        lambda: StochasticBlock(p_in=2.0),
    ],
)
def test_invalid_params_raise(ctor):
    with pytest.raises(ValueError):
        ctor()


def _params(topo):
    cfg = topo.to_config()
    cfg.pop("name")
    return cfg
