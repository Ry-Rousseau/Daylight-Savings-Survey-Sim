# Status

**Phase:** Phase 1 — Memory ✅ **complete** (DoD met) + model-capability addendum done (ADR 0005). Next: Phase 2 — Game Master / interaction.
**Updated:** 2026-07-20

## Now
Phase 1 built and green on branch **`phase-1-memory`** (uncommitted, awaiting your review). Per-agent memory: `embeddings` (BGE-small on CUDA), `memory` (numpy `MemoryStore`, Park recency/importance/relevance, `RetrievalConfig`), `prompts` (all stage prompts centralized), `importance` (pluggable, authored + LLM poignancy), `memory_seeds`, `questions` (canonical annotated `DST_QUESTION`). `Agent` now retrieves → injects → **unchanged** `choose()` → **R19** writeback. 13 tests pass (8 new deterministic). **DoD: same persona, opposite seeded memories → P(permanent-DST) delta +1.000, empty control on neither pole** (`notebooks/experiments/phase1_memory_dod.ipynb`, K=20). Checkpoint: `docs/checkpoints/phase-1.md`. ADRs: `0004` (memory stack). torch cu124 confirmed `cuda True` on the RTX 4070 Ti SUPER.

## Model-capability addendum — done (ADR 0005)
Swept the two-agent DoD across same-family **Qwen3 8b → 14b → 32b** (no Qwen3-72B exists), K=15, annotated + plain wording. Memory delta = 1.0 at all sizes (annotated); the option annotation is an **8B-only crutch** (plain-wording delta: 8b 0.0, 14b/32b 1.0); larger model *less* collapsed at baseline (weak single-persona proxy — real convergence test is P4/P5). **Baseline revised to `qwen/qwen3-32b`** (`llm.py:DEFAULT_MODEL`; amends ADR 0002; P5 self-host → Qwen3-32B-AWQ). Evidence: `notebooks/experiments/phase1_model_sweep.ipynb` + `data/phase1_model_sweep.csv`. Drop-back to 14B if P6 cost/latency bites.

## Next unit — Phase 2 (Game Master / interaction)
Symbolic action-resolution + world-state store separate from agent memory (R2/R3), the `Simulation`/`Population` container + custom tick loop, durable run-log substrate (R15/R17 foundation) with decision provenance (R29), and the simultaneous-vs-sequential within-tick update scheme (R28). Design authored ahead in `docs/design/run-architecture.md`; firm storage-tech + tick model into ADRs at P2. See `PHASE_PLAN.md` Phase 2.

## Open threads
- **Run-level architecture (scope before P2):** persistence/logging substrate (R15/R17 assume it; nothing durable today), the `Simulation`/`Population` container + custom tick loop, and **retrieval provenance** for interpretability (we log the self-reported `reason` but not the scored memory set that drove an answer). Designed in `docs/design/run-architecture.md`; firm into ADRs at P2. Standing call: retrieval store stays in-RAM numpy (persistence is additive, no global vector index — R2).
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
