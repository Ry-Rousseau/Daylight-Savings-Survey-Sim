"""Game Master resolution tests — deterministic, no network (R24/R25)."""
from polis.actions import Action, ActionType, MemoryWrite, WorldUpdate
from polis.game_master import GameMaster
from polis.memory import KIND_HEARD

GM = GameMaster()
STANCE = "Adopt permanent daylight saving time"


def test_speak_writes_to_listeners_not_self():
    action = Action.speak(STANCE, "Later sunsets would help my business.")
    effects = GM.resolve(action, actor_label="a rooftop bar owner", neighbors=["b", "c"], now=1.0)
    writes = [e for e in effects if isinstance(e, MemoryWrite)]
    assert {w.target_agent_id for w in writes} == {"b", "c"}  # never the speaker
    assert all(w.kind == KIND_HEARD for w in writes)
    assert all("a rooftop bar owner said:" in w.text for w in writes)
    assert all(w.created_at == 1.0 for w in writes)


def test_speak_emits_one_world_update():
    action = Action.speak(STANCE, "Later sunsets.")
    effects = GM.resolve(action, actor_label="owner", neighbors=["b"], now=0.0)
    updates = [e for e in effects if isinstance(e, WorldUpdate)]
    assert len(updates) == 1 and updates[0].stance == STANCE


def test_abstain_is_a_noop():
    effects = GM.resolve(Action.abstain(), actor_label="owner", neighbors=["b"], now=0.0)
    assert effects == []  # R25


def test_malformed_speak_degrades_to_noop():
    # SPEAK with no stance/utterance must not corrupt the stream.
    bad = Action(action_type=ActionType.SPEAK)
    assert GM.resolve(bad, actor_label="owner", neighbors=["b"], now=0.0) == []


def test_speak_to_no_neighbors_still_updates_world():
    # An isolated speaker reaches nobody but its expressed stance still counts.
    effects = GM.resolve(Action.speak(STANCE, "hi"), actor_label="owner", neighbors=[], now=0.0)
    assert [type(e) for e in effects] == [WorldUpdate]
