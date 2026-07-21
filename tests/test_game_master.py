"""Game Master resolution tests — deterministic, no network (R24/R25)."""
from polis.actions import (
    ACTION_SPACE_VERSION,
    Action,
    ActionType,
    MemoryWrite,
    WorldUpdate,
    action_json_schema,
)
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


# --- SHARE_CONSIDERATION: reasons, not votes (ADR 0017, R23/R24/R25) ----------

CONSIDERATION = "As a night-shift nurse, my evenings are when I'm finally alive."


def test_consideration_writes_to_listeners_with_no_world_update():
    # A consideration reaches neighbours as a heard memory but emits NO WorldUpdate,
    # so it never touches the stance tally (the key design point).
    action = Action.consider(CONSIDERATION)
    effects = GM.resolve(action, actor_label="a night-shift nurse", neighbors=["b", "c"], now=2.0)
    writes = [e for e in effects if isinstance(e, MemoryWrite)]
    assert {w.target_agent_id for w in writes} == {"b", "c"}  # never the sharer
    assert all(w.kind == KIND_HEARD for w in writes)
    assert all("a night-shift nurse shared:" in w.text for w in writes)
    assert all(w.created_at == 2.0 for w in writes)
    assert [e for e in effects if isinstance(e, WorldUpdate)] == []  # no vote → no tally


def test_malformed_consideration_degrades_to_noop():
    # SHARE_CONSIDERATION with no text must not corrupt the stream (like a bad SPEAK).
    bad = Action(action_type=ActionType.SHARE_CONSIDERATION)
    assert GM.resolve(bad, actor_label="owner", neighbors=["b"], now=0.0) == []


def test_consider_helper_and_validity():
    a = Action.consider(CONSIDERATION)
    assert a.action_type is ActionType.SHARE_CONSIDERATION
    assert a.stance is None and a.utterance is None
    assert a.is_valid_consideration() and not a.is_valid_speak()
    assert not Action(action_type=ActionType.SHARE_CONSIDERATION).is_valid_consideration()


def test_action_space_version_bumped_and_schema_permits_consideration():
    assert ACTION_SPACE_VERSION == 3
    schema = action_json_schema([STANCE])
    assert "share_consideration" in schema["properties"]["action_type"]["enum"]
    assert schema["properties"]["consideration"] == {"type": "string"}
    assert schema["required"] == ["action_type"]  # consideration GM-validated, not required


# --- REBUT: active pushback that still states a position (ADR 0018, R23/R24/R11) --

def test_rebut_writes_pushback_and_emits_world_update():
    # A rebut reaches neighbours as a "pushed back" memory AND emits a WorldUpdate
    # (it still states a position — unlike a stanceless consideration).
    action = Action.rebut(STANCE, "Earlier sunrises ignore everyone who works evenings.")
    effects = GM.resolve(action, actor_label="a bartender", neighbors=["b", "c"], now=3.0)
    writes = [e for e in effects if isinstance(e, MemoryWrite)]
    assert {w.target_agent_id for w in writes} == {"b", "c"}
    assert all(w.kind == KIND_HEARD and "a bartender pushed back:" in w.text for w in writes)
    updates = [e for e in effects if isinstance(e, WorldUpdate)]
    assert len(updates) == 1 and updates[0].stance == STANCE


def test_malformed_rebut_degrades_to_noop():
    assert GM.resolve(Action(action_type=ActionType.REBUT), actor_label="x", neighbors=["b"], now=0.0) == []


def test_rebut_helper_and_validity():
    a = Action.rebut(STANCE, "no")
    assert a.action_type is ActionType.REBUT and a.stance == STANCE
    assert a.is_valid_rebut() and not a.is_valid_speak() and not a.is_valid_consideration()
    assert not Action(action_type=ActionType.REBUT, stance=STANCE).is_valid_rebut()  # needs utterance
    assert "rebut" in action_json_schema([STANCE])["properties"]["action_type"]["enum"]
