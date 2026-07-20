# ADR 0006: Run-log storage — SQLite

Status: accepted
Phase: 2

## Spike question

What storage tech backs the durable tier-3 run log (the R15/R17 foundation) —
the append-only event stream of actions, effects, per-tick metrics, and decision
provenance that offline validation and interpretability will query?

## Context

`docs/design/run-architecture.md` names three state tiers; tier 3 (the durable
run log) is the one no prior phase built. R15 wants metrics logged *per tick*, R17
wants every run versioned against its config so an observed convergence is
traceable, R29 wants each decision's retrieved-memory set recorded, and P7 wants
surveys *over simulated time* — all assume durable, queryable state. Today
everything is ephemeral RAM. The log is **write-only during a run** and
**queried offline**; the hot retrieval path stays in-RAM numpy (ADR 0004) and is
explicitly *not* what this decision is about — durability + analytical
queryability is the need, not retrieval speed.

## Options considered

1. **SQLite** — single file, SQL-queryable, `sqlite3` in the stdlib (no new dep),
   durable across processes, indexable. Writing is a serial `INSERT`; the tick
   loop is single-threaded so that's fine. Slightly more ceremony than appending
   text.
2. **JSONL** — dead-simple append-only, human-readable, git-diffable. But every
   analytical query (P5 divergence trajectory, R29 provenance tracing) must load
   and parse the whole file; no indexed/filtered queries, no run/event join.
3. **Parquet** — columnar, ideal for later pandas-heavy analysis. But append is
   awkward (batch/rewrite semantics), it adds a `pyarrow` dependency, and it's
   not human-readable for quick debugging — premature for a substrate whose
   query patterns aren't settled yet.

## Decision

**SQLite**, via `src/polis/runlog.py` (`RunLog`). Two tables: `runs(run_id,
config_json, config_hash, started_at)` and `events(event_id, run_id, tick,
agent_id, event_type, payload_json)`, indexed on `(run_id, event_id)`. Event
vocabulary is a closed set: `tick_marker`, `action`, `memory_write`,
`world_update`, `retrieval`. `config_hash` is a stable SHA-256 over the
canonicalized config (R17).

## Why

The tier-3 need is durability + *queryability* for offline validation and
provenance, not append throughput. SQLite gives filtered/joined queries (e.g.
"all `retrieval` events for agent X in run R") with zero new dependencies and
single-file durability, which JSONL can't without loading everything and parquet
can't append cleanly. It's the lean the design note already pointed at. If P5
analysis turns out columnar-heavy, exporting a SQLite run to parquet for analysis
is trivial and additive — the reverse (starting in parquet, needing ad-hoc
queries mid-run) is not.

## Consequences

- Locks the run log to a relational, append-only shape; a new event kind is a new
  `event_type` value, not a schema migration (payloads are JSON).
- Writes are serialized by the single-threaded tick loop; the LangGraph survey
  fan-out does **not** write here (it's the query layer, R22), so no concurrent
  writers. Revisit (WAL / per-worker connections) only if a future phase makes
  the *log* writer concurrent.
- Revisit toward parquet **export** if P5 divergence analysis is columnar-heavy —
  as an additive analysis step, not a substrate change.

## Rules touched

R15 (per-tick logging substrate), R17 (config-versioned runs), R29 (decision
provenance stream). Foundation for R14 (metrics computed offline at P5).
