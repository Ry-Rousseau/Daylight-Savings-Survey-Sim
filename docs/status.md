# Status

**Phase:** Phase 2 — Game Master / interaction ✅ **complete** (DoD met). Next: Phase 3 — Scheduling & throughput.
**Updated:** 2026-07-20

## Now
Phase 2 built and green on branch **`phase-2-game-master`** (uncommitted, awaiting your review). The Game-Master / interaction skeleton + run-level scaffolding: `runlog` (append-only SQLite tier-3 log, config hash R17), `world` (tier-2 `WorldState` + read-only `WorldView`, R2/R3), `actions` (closed versioned SPEAK/ABSTAIN space R23, `Effect`/`RetrievalProvenance`), `game_master` (deterministic resolution R24, abstain no-op R25), `simulation` (`Population`/`Simulation`/`Run` + **custom tick loop** R22, R28 update scheme). `memory` gained `score_components`/`retrieve_scored` (R29) + `KIND_HEARD`; `Agent.act()` added; `llm.decide()` shares a helper with an unchanged `choose()`. **33 deterministic tests pass** (20 new) + live `choose()` green. **DoD: 2 seeded agents run 3 ticks; SPEAK resolves into the listener's memory + world tally; SQLite log records actions/effects/R29 provenance and reopens from disk consistently** (`notebooks/experiments/phase2_interaction_dod.ipynb`). Checkpoint: `docs/checkpoints/phase-2.md`. ADRs: `0006` (SQLite), `0007` (simultaneous update default, R28), `0008` (action space v1).

**Phase 1** remains complete on `phase-1-memory` (merged to `main`): per-agent memory, retrieve → inject → **unchanged** `choose()` → R19 writeback; DoD P(permanent-DST) delta +1.000. Checkpoint `docs/checkpoints/phase-1.md`; ADRs `0004`/`0005`. torch cu124 `cuda True` on the RTX 4070 Ti SUPER.

## Model-capability addendum — done (ADR 0005)
Swept the two-agent DoD across same-family **Qwen3 8b → 14b → 32b** (no Qwen3-72B exists), K=15, annotated + plain wording. Memory delta = 1.0 at all sizes (annotated); the option annotation is an **8B-only crutch** (plain-wording delta: 8b 0.0, 14b/32b 1.0); larger model *less* collapsed at baseline (weak single-persona proxy — real convergence test is P4/P5). **Baseline revised to `qwen/qwen3-32b`** (`llm.py:DEFAULT_MODEL`; amends ADR 0002; P5 self-host → Qwen3-32B-AWQ). Evidence: `notebooks/experiments/phase1_model_sweep.ipynb` + `data/phase1_model_sweep.csv`. Drop-back to 14B if P6 cost/latency bites.

## Next unit — Phase 3 (Scheduling & throughput)
Warm/cold scheduler + batched local inference (R5/R6); benchmark 20–30 agents over multiple ticks with logged latency/cost. The P2 tick loop is single-threaded and calls the endpoint once per agent per tick — this is where batching lands. **Watch:** the run log's serial-writer assumption (ADR 0006) holds only while the *log* writer stays single-threaded; revisit if P3 parallelizes logging. See `PHASE_PLAN.md` Phase 3.

## Open threads
- **Run-level architecture — RESOLVED at P2.** Durable run log (SQLite, ADR 0006), `Simulation`/`Population` container + custom tick loop, world-state store (R2/R3), and R29 retrieval provenance are all built and green. Standing call held: retrieval store stays in-RAM numpy; persistence is additive, no global vector index (R2).
- **R27 action-space adequacy (carry to P4/P5):** the P2 action space is deliberately narrow (SPEAK/ABSTAIN); adequacy must be re-checked *separately* from any homogeneity metric before convergence results are read, since a narrow action space can suppress observable divergence the R16 null baseline won't catch (ADR 0008).
- **World tally not yet read by agents:** `WorldState.stance_tally` is logged (R3) but agents don't condition on it yet — turning it into a live consensus pressure is a deliberate P4 dynamics choice.
- **Push:** `origin` set (`github.com/Ry-Rousseau/NYC-Daylight-Savings-Sim`); `main` up to date. `phase-2-game-master` branch is local/uncommitted, awaiting review.
- **Endpoint:** phased (ADR 0002/0005) — **OpenRouter** `qwen/qwen3-32b` now (P0–2 baseline revised to 32B), self-hosted vLLM at P5+. Details: `docs/query_handbook.md`. Local 4070Ti reserved for lightweight local models (e.g. P1 embeddings).
- **Census dataset:** working choice = **ACS PUMS** for NYC demographic seeds (confirmed direction; finalize field mapping at Phase 5).
- **Validation ground truth:** a published DST opinion figure to calibrate against — to source at Phase 5/7. (R21 calibration hold-out is intentionally out of current scope.)
- **viz_theme:** categoricals deferred until the persona schema firms up (~Phase 5).

## Resolved this session
- **Env:** Python **3.11** via the project `.venv` — no reconciliation needed; run via `.venv/Scripts/python.exe`.
- **`.venv` setup:** `.venv` was created by `uv venv` with no `pip` and no registered Jupyter kernel (the only kernel on the machine pointed at an unrelated Python 3.13 install). Fixed: installed `pip` + the Phase 0 baseline via `uv pip install`; registered a `polis` kernel (`ipykernel install --user --name polis`) whose kernel.json points at the `.venv` interpreter by absolute path. Notebooks must select the "Python 3.11 (polis .venv)" kernel, not the default "Python 3 (ipykernel)" one.
- **`pyproject.toml` (brief task 1):** already existed on disk (with `src/polis` + `tests/test_smoke.py`, `pytest` green) but was **untracked** — not yet `git add`ed from an earlier session. Added a `notebooks` optional-dependency group (`ipykernel`, `apify-client`) so `uv pip install -e ".[dev,notebooks]"` reproduces the working env from a fresh clone instead of silently missing the packages `notebooks/engineering/twitter_apify_toolkit.ipynb` needs. README quickstart updated to match. `pyproject.toml`/`tests/test_smoke.py` still need `git add` + commit — left for the user (commits are the user's).
- **`twitter_apify_toolkit.ipynb`:** cleared the stale `ModuleNotFoundError` traceback baked into its outputs (artifact of the missing-pip issue) and pinned its kernelspec metadata to `polis` so it opens with the right kernel automatically instead of prompting/defaulting to the stray 3.13 one.
- **Rules:** R18–R20, R22–R27 defined in `ARCHITECTURE.md` (R23/R24 action space + Game Master resolution, R25 abstain, R26 topology-mutation logging, R27 action-space adequacy); R21 dropped from scope.
- **Doc reconciliation:** architecture rules → `docs/ARCHITECTURE.md`; phase plan → `docs/PHASE_PLAN.md`; decisions → numbered ADRs in `docs/adr/` with `docs/adr/template.md`; session handoffs live here in `status.md`. `conventions.md` updated to match.
- **Phase 0 built:** `src/polis/` (survey/llm/persona/agent/graph/__main__), 5 tests green (1 live), `python -m polis` works. Endpoint = OpenRouter (ADR 0002); Qwen3 reasoning off + soft json_schema validate/retry (ADR 0003).

## Handoff format
On a state-changing session end: update this file — what changed, what's live vs half-done, decisions + why, blockers. Any decision gets an ADR in `docs/adr/` (format: `docs/adr/template.md`).
