"""Durable run log — the tier-3 append-only event stream (R15/R17 foundation; R29).

Tier 3 in ``docs/design/run-architecture.md``: write-only during a run, queried
offline for validation and interpretability. SQLite (ADR 0006) — single-file,
SQL-queryable, stdlib, durable across processes; the substrate the P5 divergence
metrics (R14/R15) and provenance analysis (R29) will run on. This phase builds
the substrate, not the metrics.

Every run is versioned against its config (R17): ``log_run`` hashes the config so
any observed convergence is traceable to the layer that produced it. Events are an
append-only stream keyed by ``run_id`` — actions, memory writes, world updates, and
per-decision retrieval provenance (R29) — reconstructable in event order offline.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

# Event-stream vocabulary. Kept as a closed set so offline queries can filter by
# type without guessing; topology-mutating actions (R26) get their own type,
# distinct from content-exchange ``action`` events.
EVENT_TICK = "tick_marker"
EVENT_TICK_METRICS = "tick_metrics"  # per-tick throughput summary (P3; R15-style trajectory)
EVENT_ACTION = "action"
EVENT_MEMORY_WRITE = "memory_write"
EVENT_WORLD_UPDATE = "world_update"
EVENT_RETRIEVAL = "retrieval"  # R29 decision provenance
EVENT_FEED = "feed_delivery"  # a deliberately shared external signal, logged so its effect is traceable (R3)
# R26 seam: tie formation/dissolution changes the graph rather than exchanging
# information over it, so it gets its own stream. Reserved at P4 (topology is static
# and swapped only *between* runs for the R13 counterfactual); tie-mutating actions
# that emit this are a later dynamics feature, not built this phase.
EVENT_TIE_CHANGE = "tie_change"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    started_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    event_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    tick        INTEGER,
    agent_id    TEXT,
    event_type  TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, event_id);
"""


def config_hash(config: dict[str, Any]) -> str:
    """Stable SHA-256 over a config dict (R17).

    Canonicalized with sorted keys so logically-equal configs hash equal
    regardless of insertion order; sensitive to any value change.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class RunLog:
    """Append-only SQLite event log. Write-only during a run; queried offline.

    Writes are serialized by the single-threaded tick loop that owns this log;
    the query-layer survey fan-out (LangGraph) does not write here.
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    @classmethod
    def open(cls, path: str) -> "RunLog":
        """Open (creating if absent) a run log at ``path``. Use a real file for
        durability; ``":memory:"`` is fine for a single-connection test."""
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return cls(conn)

    def log_run(self, config: dict[str, Any]) -> str:
        """Register a run and its versioned config (R17); returns a new run_id."""
        run_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO runs (run_id, config_json, config_hash, started_at) VALUES (?, ?, ?, ?)",
            (
                run_id,
                json.dumps(config, sort_keys=True, default=str),
                config_hash(config),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()
        return run_id

    def log_event(
        self,
        run_id: str,
        *,
        event_type: str,
        payload: dict[str, Any],
        tick: int | None = None,
        agent_id: str | None = None,
    ) -> int:
        """Append one event to the stream; returns its monotonic event_id."""
        cur = self._conn.execute(
            "INSERT INTO events (run_id, tick, agent_id, event_type, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, tick, agent_id, event_type, json.dumps(payload, default=str)),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def events(
        self, run_id: str, *, event_type: str | None = None
    ) -> list[dict[str, Any]]:
        """Return the run's events in append order, payloads parsed back to dicts."""
        sql = "SELECT event_id, run_id, tick, agent_id, event_type, payload_json FROM events WHERE run_id = ?"
        params: list[Any] = [run_id]
        if event_type is not None:
            sql += " AND event_type = ?"
            params.append(event_type)
        sql += " ORDER BY event_id"
        rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "event_id": r["event_id"],
                "run_id": r["run_id"],
                "tick": r["tick"],
                "agent_id": r["agent_id"],
                "event_type": r["event_type"],
                "payload": json.loads(r["payload_json"]),
            }
            for r in rows
        ]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return the run's config record, or None if unknown."""
        r = self._conn.execute(
            "SELECT run_id, config_json, config_hash, started_at FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if r is None:
            return None
        return {
            "run_id": r["run_id"],
            "config": json.loads(r["config_json"]),
            "config_hash": r["config_hash"],
            "started_at": r["started_at"],
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
