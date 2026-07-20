"""Durable run-log tests — no network. A temp file exercises real cross-connection
durability (R15/R17); an in-memory DB can't, since each connection is its own db.
"""
from polis.runlog import (
    EVENT_ACTION,
    EVENT_RETRIEVAL,
    RunLog,
    config_hash,
)


def test_config_hash_is_order_independent():
    a = config_hash({"model": "qwen", "seed": 7, "scheme": "simultaneous"})
    b = config_hash({"scheme": "simultaneous", "seed": 7, "model": "qwen"})
    assert a == b


def test_config_hash_is_value_sensitive():
    base = {"model": "qwen", "seed": 7}
    assert config_hash(base) != config_hash({**base, "seed": 8})


def test_events_return_in_append_order(tmp_path):
    log = RunLog.open(str(tmp_path / "run.db"))
    run_id = log.log_run({"model": "qwen", "scheme": "simultaneous"})
    log.log_event(run_id, event_type=EVENT_ACTION, payload={"n": 1}, tick=0, agent_id="a")
    log.log_event(run_id, event_type=EVENT_ACTION, payload={"n": 2}, tick=0, agent_id="b")
    evs = log.events(run_id)
    assert [e["payload"]["n"] for e in evs] == [1, 2]
    assert evs[0]["event_id"] < evs[1]["event_id"]
    log.close()


def test_event_type_filter(tmp_path):
    log = RunLog.open(str(tmp_path / "run.db"))
    run_id = log.log_run({"model": "qwen"})
    log.log_event(run_id, event_type=EVENT_ACTION, payload={"k": "act"})
    log.log_event(run_id, event_type=EVENT_RETRIEVAL, payload={"k": "prov"})
    only = log.events(run_id, event_type=EVENT_RETRIEVAL)
    assert len(only) == 1 and only[0]["payload"]["k"] == "prov"
    log.close()


def test_durable_across_reopen(tmp_path):
    """Write, close, reopen a fresh connection to the same file: stream survives."""
    path = str(tmp_path / "run.db")
    log = RunLog.open(path)
    run_id = log.log_run({"model": "qwen", "seed": 1})
    log.log_event(run_id, event_type=EVENT_RETRIEVAL, payload={"memories": ["m1", "m2"]},
                  tick=3, agent_id="nurse")
    log.close()

    reopened = RunLog.open(path)
    run = reopened.get_run(run_id)
    assert run is not None and run["config"]["seed"] == 1
    assert run["config_hash"] == config_hash({"model": "qwen", "seed": 1})
    evs = reopened.events(run_id)
    assert len(evs) == 1
    assert evs[0]["payload"]["memories"] == ["m1", "m2"]
    assert evs[0]["tick"] == 3 and evs[0]["agent_id"] == "nurse"
    reopened.close()
