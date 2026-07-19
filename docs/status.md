# Status

**Phase:** Phase 0 — walking skeleton ✅ **complete**. Next: Phase 1 — Memory.
**Updated:** 2026-07-19

## Now
Phase 0 built and green on `main` (no branch, per your call). `src/polis/` = `survey`, `llm` (OpenRouter client), `persona`, `agent`, `graph` (LangGraph fan-out/gather), `__main__`. `python -m polis` surveys 3 personas on DST → 3 schema-valid answers. 5 tests pass (1 live). Endpoint phased (ADR 0002). Checkpoint: `docs/checkpoints/phase-0.md`. Large uncommitted set on `main` awaiting your commit.

## Next unit — Phase 1 (Memory)
DoD: two agents with different **seeded memories** give measurably different survey answers. Add a per-agent memory store + recency/importance/relevance retrieval the agent reads before `choose()`. See `PHASE_PLAN.md` Phase 1. Spikes: embedding model (candidate: local on the 4070Ti), vector store, scoring/decay.

## Open threads
- **Push:** `origin` now set (`github.com/Ry-Rousseau/NYC-Daylight-Savings-Sim`); branch is up to date with `origin/main`. No longer blocked.
- **Endpoint:** phased (ADR 0002) — **OpenRouter** `qwen/qwen3-8b` now (P0–2), self-hosted vLLM at P5+. Details: `docs/query_handbook.md`. Local 4070Ti reserved for lightweight local models (e.g. P1 embeddings).
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
