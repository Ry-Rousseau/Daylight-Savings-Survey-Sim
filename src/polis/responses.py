"""Read canonical responses out of a run log into tidy DataFrames (analysis surface).

The engine's write path (``RunLog``) returns opaque event rows keyed by run_id; this
is the offline read complement — flatten a run's ``action`` / ``retrieval`` events into
pandas frames a notebook can query directly. It sits beside ``metrics`` (divergence
math): this is the row-level "what did each agent answer each tick, and what memory
drove it (R29)" view — the hands-on surface for inspecting the simulation's responses.

Layer-1 (the individual, no-interaction survey) is a separate cross-section written to
CSV by the survey runner; this module is the Layer-2 (simulation) reader.
"""
from __future__ import annotations

import re

import pandas as pd

from .actions import ActionType
from .memory import KIND_HEARD
from .runlog import EVENT_ACTION, EVENT_RETRIEVAL, RunLog

# A stance is expressed only by SPEAK or REBUT (a REBUT is a SPEAK framed as pushback);
# SHARE_CONSIDERATION carries a reason but no stance, ABSTAIN nothing. Matches
# ``metrics.latest_speaks`` so the two readers agree on what counts as a position.
_STANCE_ACTIONS = {ActionType.SPEAK.value, ActionType.REBUT.value}


class BoundRun:
    """A ``RunLog`` pinned to one run_id, exposing the ``.events(event_type=...)`` shape
    the readers (here and in ``metrics``) expect. The offline entry point for an
    existing ``.sqlite`` on disk, where there is no live ``Run`` object to hand."""

    def __init__(self, log: RunLog, run_id: str):
        self.log = log
        self.run_id = run_id

    def events(self, **kw):
        return self.log.events(self.run_id, **kw)


def open_run(path: str, run_id: str | None = None) -> BoundRun:
    """Open an existing run log for analysis, bound to one run.

    ``run_id`` defaults to the most recent run in the file (``RunLog.list_runs`` is
    ordered newest-first) — the common case, since most result ``.sqlite`` files hold a
    single run. Plugs straight into ``metrics`` functions as well as the frame builders
    below."""
    log = RunLog.open(path)
    if run_id is None:
        runs = log.list_runs()
        if not runs:
            raise ValueError(f"no runs in {path}")
        run_id = runs[0]["run_id"]
    return BoundRun(log, run_id)


_ACTION_COLS = [
    "tick", "agent_id", "action_type", "stance", "utterance", "consideration", "model",
]


def actions_frame(run) -> pd.DataFrame:
    """One row per ``action`` event: what each agent did each tick. ``stance`` is the
    position taken (SPEAK/REBUT) or NaN; ``utterance`` is the free text — the response
    voice that Layer-1's stances CSV discards. Empty frame with the fixed columns if the
    run logged no actions (a stub-client survey run)."""
    rows = [
        {
            "tick": e["tick"],
            "agent_id": e["agent_id"],
            "action_type": e["payload"].get("action_type"),
            "stance": e["payload"].get("stance"),
            "utterance": e["payload"].get("utterance"),
            "consideration": e["payload"].get("consideration"),
            "model": e["payload"].get("model"),
        }
        for e in run.events(event_type=EVENT_ACTION)
    ]
    return pd.DataFrame(rows, columns=_ACTION_COLS)


_RETRIEVAL_COLS = [
    "tick", "agent_id", "rank", "query", "text", "kind",
    "created_at", "recency", "importance", "relevance", "total",
]


def retrieval_frame(run) -> pd.DataFrame:
    """One row per retrieved memory (R29 provenance), flattened from the per-decision
    ``retrieval`` events: which memories the agent pulled in to decide, each with its
    recency / importance / relevance / total scores. ``rank`` is retrieval order within
    the decision. Join to ``actions_frame`` / ``flips`` on ``(agent_id, tick)`` to see
    *what an agent had just heard* when it took (or changed) its stance."""
    rows = []
    for e in run.events(event_type=EVENT_RETRIEVAL):
        p = e["payload"]
        for rank, h in enumerate(p.get("hits", [])):
            rows.append(
                {
                    "tick": e["tick"],
                    "agent_id": e["agent_id"],
                    "rank": rank,
                    "query": p.get("query"),
                    "text": h.get("text"),
                    "kind": h.get("kind"),
                    "created_at": h.get("created_at"),
                    "recency": h.get("recency"),
                    "importance": h.get("importance"),
                    "relevance": h.get("relevance"),
                    "total": h.get("total"),
                }
            )
    return pd.DataFrame(rows, columns=_RETRIEVAL_COLS)


def stance_pivot(actions: pd.DataFrame) -> pd.DataFrame:
    """Wide agent x tick stance table (stance-expressing actions only), forward-filled
    along ticks so a tick where an agent only shared/abstained carries its last
    expressed stance. Rows are agents that expressed a stance at least once; columns are
    ticks in ascending order. The at-a-glance "who held what, when" view."""
    st = actions[
        actions["action_type"].isin(_STANCE_ACTIONS) & actions["stance"].notna()
    ]
    if st.empty:
        return pd.DataFrame()
    wide = st.pivot_table(
        index="agent_id", columns="tick", values="stance", aggfunc="last"
    )
    return wide.sort_index(axis=1).ffill(axis=1)


_FLIP_COLS = ["agent_id", "tick", "from_stance", "to_stance"]


def flips(actions: pd.DataFrame) -> pd.DataFrame:
    """One row per stance change: the tick an agent moved and from/to which stance.

    The seed for diagnosing over-influence ("personas don't stick to their guns"): join
    to ``retrieval_frame`` on ``(agent_id, tick)`` to read what the agent had just heard
    at the moment it changed its mind, and to ``actions_frame`` for the utterance it gave
    when it flipped."""
    wide = stance_pivot(actions)
    rows = []
    for aid, series in wide.iterrows():
        prev = None
        for tick, val in series.items():
            if pd.isna(val):
                continue
            if prev is not None and val != prev:
                rows.append(
                    {"agent_id": aid, "tick": tick, "from_stance": prev, "to_stance": val}
                )
            prev = val
    return pd.DataFrame(rows, columns=_FLIP_COLS)


# Social-proof / agreement language — the fingerprint of a sycophantic flip (joining the
# room) rather than an argument-driven one. A heuristic flag, not ground truth: it is a
# lower bound (misses paraphrases), read alongside the utterance itself, not instead of it.
_SOCIAL_RE = re.compile(
    r"convinc|persuad|fair point|good point|you.?re right|everyone|the others|the ladies|"
    r"the folks|the fellas|the neighbors|hearing|consensus|\bagree|majority|made me|"
    r"changed my|come around|see (?:your|the) point|swayed|both sides|i'?m with|side with|"
    r"add my voice|quoting me",
    re.I,
)

_FLIP_REPORT_COLS = _FLIP_COLS + ["utterance", "n_heard", "references_others"]


def flip_report(run) -> pd.DataFrame:
    """Every stance flip joined to the utterance the agent gave *at the flip* and how many
    ``heard`` (peer) memories it had retrieved that tick — the over-influence diagnostic.

    ``n_heard`` measures the peer pressure in context; ``references_others`` flags social-
    proof language in the flip utterance (see ``_SOCIAL_RE``). A run where flips carry high
    ``n_heard`` and mostly ``references_others`` is flipping by conformity, not argument —
    the signal the persistence / heard-downweight levers aim to move. Empty (with the full
    columns) if the run had no flips."""
    acts = actions_frame(run)
    fl = flips(acts)
    if fl.empty:
        return pd.DataFrame(columns=_FLIP_REPORT_COLS)
    utt_by_key = dict(
        zip(zip(acts["agent_id"], acts["tick"]), acts["utterance"])
    )
    fl["utterance"] = [utt_by_key.get((a, t)) for a, t in zip(fl["agent_id"], fl["tick"])]

    prov = retrieval_frame(run)
    heard = prov[prov["kind"] == KIND_HEARD]
    heard_counts = heard.groupby(["agent_id", "tick"]).size() if not heard.empty else None
    fl["n_heard"] = [
        int(heard_counts.get((a, t), 0)) if heard_counts is not None else 0
        for a, t in zip(fl["agent_id"], fl["tick"])
    ]
    fl["references_others"] = (
        fl["utterance"].fillna("").str.contains(_SOCIAL_RE)
    )
    return fl[_FLIP_REPORT_COLS]
