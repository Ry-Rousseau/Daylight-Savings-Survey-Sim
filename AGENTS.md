# polis — AGENTS.md

**Project:** `polis` builds a surveyable "silicon sample" of New York City — ~100 census-seeded LLM personas we poll with single-select questions on **daylight saving time** to estimate NYC opinion, treating opinion **convergence** (loss of realistic diversity) as a layered, cross-cutting risk rather than an end-of-project patch.

## Routing — read in this order
- `docs/conventions.md` — house style, the single source of truth for *how* we work. **Read first, every session.**
- `docs/status.md` — current phase, next unit, open threads. **Read at session start.**
- `docs/design/polis-object.md` — what the object is + its interface. `docs/ARCHITECTURE.md` — the **R-number architecture rules** (R1–R20 + R22, across five layers): the standing spec of what must be true regardless of phase. (R21, a calibration hold-out, is intentionally out of current scope.)
- `docs/PHASE_PLAN.md` — **the authoritative phase plan**: 8 phases (walking skeleton → full survey), each with spike question(s), DoD, layers touched, rules activated.
- `docs/adr/` — ADRs: why a choice was made, options weighed, rules touched (`docs/adr/template.md` is the ADR format). Don't re-litigate.
- `docs/briefs/` — the current unit's brief. `docs/checkpoints/` — phase-gate notes.
- Background: `docs/project_brief.md` (direction) · `docs/requirements_features.md` (subsystems under the prompt).

## The object
`src/polis/` is the deliverable — a source-first opinion-simulation engine. Headline ops: `Population.from_census(...)` → seed; `population.survey(question)` → answer distribution; `Simulation(pop, topology).run(ticks)` → per-tick logged trajectory + divergence metric. Notebooks import *down* from `src/`; they never redefine its logic. Orchestration uses two tools (R22): a **custom tick loop** for the simulation core, and **LangGraph** for the bounded query/survey fan-out/gather. Structured output via vLLM grammar-constrained decoding.

## Inputs / data sources
- **LLM endpoint (persona backend):** phased (ADR 0002). **Now (P0–2):** managed **OpenRouter**, OpenAI-compatible at `https://openrouter.ai/api/v1`, model `qwen/qwen3-32b` (baseline set by ADR 0005; 8B needed hand-annotated survey options, 14B/32B do not), key `OPENROUTER_API_KEY` in `.env`. **P5+:** self-hosted **vLLM** (`Qwen/Qwen3-32B-AWQ`, RunPod/Koyeb) with pinned commit + quantization, where R6/R17 rigor is load-bearing. Model pinned & logged per call (R6). Client: `src/polis/llm.py`; query examples + params: `docs/query_handbook.md`. Note: disable Qwen3 reasoning for survey answers; OpenRouter json_schema is soft-enforced (validate+retry), hard grammar comes with vLLM. The local RTX 4070Ti (16 GB) stays free for lightweight local models (e.g. P1 embeddings).
- **Persona seeds (census):** NYC demographics — occupation, voting-age band, sex, race, industry, education (+ family circumstances if available). *Dataset TBD — to confirm (likely ACS PUMS for NYC PUMAs).*
- **Opinion seeds (optional, P2+):** DST posts scraped from X. Optional/deferred — ToS-sensitive.
- **Validation ground truth:** a published DST opinion figure for NYC/US to calibrate against. *Source TBD — to confirm.*

## Environment
- Python **3.11** via the project virtualenv `.venv` (pinned in `.python-version`). Run with `.venv/Scripts/python.exe` or activate `.venv`; don't use the bare `py`/`python` (that resolves to a different 3.12 / the Windows Store stub).

## Working rules (from conventions.md)
- **Commits are the user's.** Never stage or commit without explicit approval; surface diffs and let the user drive. Working directly on `main` is fine (the user's chosen workflow) — no per-unit branch required — but the commit itself is always the user's call.
- Source-first: a notebook that needs a capability means adding it to `src/` with a test, then calling it. Version via git tags, not `_V2` filenames.
- Keep sessions scoped; end a state-changing session by updating `status.md` and writing the handoff.
