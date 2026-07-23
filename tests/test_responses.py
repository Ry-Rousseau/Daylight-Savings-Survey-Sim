"""Response-inspection tests — flatten a run log into tidy frames, no network.

Builds a small run by hand (two agents, three ticks, one stance flip) so the frame
shapes, the forward-fill, and the flip detection are exercised against a known log.
"""
import pandas as pd

from polis.responses import (
    actions_frame,
    flip_report,
    flips,
    open_run,
    retrieval_frame,
    stance_pivot,
)
from polis.runlog import EVENT_ACTION, EVENT_RETRIEVAL, RunLog


def _seed_log(path):
    """A 2-agent, 3-tick run: 'a' holds standard throughout; 'b' speaks DST at tick 0,
    only shares at tick 1 (no stance), then flips to standard at tick 2."""
    log = RunLog.open(path)
    run_id = log.log_run({"model": "test", "ticks": 3})

    def act(tick, agent, atype, stance=None, utt=None):
        log.log_event(
            run_id, event_type=EVENT_ACTION, tick=tick, agent_id=agent,
            payload={"action_type": atype, "stance": stance, "utterance": utt},
        )

    act(0, "a", "speak", "standard", "clocks should stop moving")
    act(0, "b", "speak", "dst", "long evenings are worth it")
    act(1, "a", "speak", "standard", "still standard")
    act(1, "b", "share_consideration", None, None)  # no stance this tick
    act(2, "a", "speak", "standard", "unchanged")
    act(2, "b", "rebut", "standard", "fine, you convinced me")  # the flip
    log.log_event(
        run_id, event_type=EVENT_RETRIEVAL, tick=2, agent_id="b",
        payload={"query": "daylight saving time", "hits": [
            {"text": "standard is better", "kind": "heard", "created_at": 1.0,
             "recency": 0.9, "importance": 1.0, "relevance": 0.8, "total": 0.85},
        ]},
    )
    log.close()
    return run_id


def test_open_run_defaults_to_latest(tmp_path):
    path = str(tmp_path / "run.db")
    run_id = _seed_log(path)
    run = open_run(path)
    assert run.run_id == run_id
    assert len(run.events(event_type=EVENT_ACTION)) == 6


def test_actions_frame_shape(tmp_path):
    path = str(tmp_path / "run.db")
    _seed_log(path)
    df = actions_frame(open_run(path))
    assert list(df.columns) == [
        "tick", "agent_id", "action_type", "stance", "utterance", "consideration", "model",
    ]
    assert len(df) == 6
    # the share_consideration row carries no stance
    share = df[df["action_type"] == "share_consideration"].iloc[0]
    assert pd.isna(share["stance"])


def test_actions_frame_empty_keeps_columns(tmp_path):
    path = str(tmp_path / "empty.db")
    log = RunLog.open(path)
    log.log_run({"model": "test"})
    log.close()
    df = actions_frame(open_run(path))
    assert df.empty and "stance" in df.columns


def test_stance_pivot_forward_fills(tmp_path):
    path = str(tmp_path / "run.db")
    _seed_log(path)
    wide = stance_pivot(actions_frame(open_run(path)))
    # tick 1 for 'b' was a share (no stance) -> forward-fill carries tick-0 dst
    assert wide.loc["b", 1] == "dst"
    assert wide.loc["b", 2] == "standard"
    assert list(wide.loc["a"]) == ["standard", "standard", "standard"]


def test_flips_detects_the_change(tmp_path):
    path = str(tmp_path / "run.db")
    _seed_log(path)
    f = flips(actions_frame(open_run(path)))
    assert len(f) == 1
    row = f.iloc[0]
    assert (row["agent_id"], row["tick"], row["from_stance"], row["to_stance"]) == (
        "b", 2, "dst", "standard",
    )


def test_retrieval_frame_flattens_hits(tmp_path):
    path = str(tmp_path / "run.db")
    _seed_log(path)
    r = retrieval_frame(open_run(path))
    assert len(r) == 1
    hit = r.iloc[0]
    assert hit["agent_id"] == "b" and hit["tick"] == 2 and hit["kind"] == "heard"
    assert hit["total"] == 0.85


def test_flip_report_joins_utterance_heard_and_social_flag(tmp_path):
    path = str(tmp_path / "run.db")
    _seed_log(path)
    rep = flip_report(open_run(path))
    assert len(rep) == 1
    row = rep.iloc[0]
    assert row["agent_id"] == "b" and row["tick"] == 2
    assert row["utterance"] == "fine, you convinced me"
    assert row["n_heard"] == 1                      # one heard memory retrieved at the flip
    assert bool(row["references_others"]) is True   # "convinced" trips the social heuristic


def test_flip_report_empty_when_no_flips(tmp_path):
    """A run where nobody changes stance returns the full columns, zero rows."""
    path = str(tmp_path / "run.db")
    log = RunLog.open(path)
    run_id = log.log_run({"model": "test"})
    log.log_event(run_id, event_type=EVENT_ACTION, tick=0, agent_id="a",
                  payload={"action_type": "speak", "stance": "standard", "utterance": "x"})
    log.close()
    rep = flip_report(open_run(path))
    assert rep.empty and "references_others" in rep.columns
