"""World-state tests — the R2/R3 write boundary. No network."""
import pytest

from polis.world import WorldState


def test_view_is_read_only_mapping():
    """An agent's view cannot mutate tier-2 state through the tally (R2)."""
    world = WorldState(roster=("a", "b"), stance_tally={"x": 1})
    view = world.view()
    with pytest.raises(TypeError):
        view.stance_tally["x"] = 99  # MappingProxyType rejects writes


def test_view_is_frozen():
    world = WorldState(roster=("a", "b"))
    view = world.view()
    with pytest.raises(Exception):
        view.tick = 5  # frozen dataclass


def test_view_is_a_snapshot_not_a_live_handle():
    """Mutating the store after handing out a view must not change that view."""
    world = WorldState(roster=("a", "b"), stance_tally={"x": 1})
    view = world.view()
    world.record_stance("x")
    assert view.stance_tally["x"] == 1  # snapshot held at 1
    assert world.stance_tally["x"] == 2  # store advanced


def test_record_stance_accumulates():
    world = WorldState(roster=("a",))
    world.record_stance("x")
    world.record_stance("x")
    world.record_stance("y")
    assert world.stance_tally == {"x": 2, "y": 1}


def test_advance_tick():
    world = WorldState(roster=("a",))
    world.advance_tick()
    assert world.tick == 1 and world.view().tick == 1
