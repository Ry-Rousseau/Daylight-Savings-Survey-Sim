# polis — CLAUDE.md

**Project:** `polis` builds a surveyable "silicon sample" of New York City — ~100 census-seeded LLM personas we poll with single-select questions on **daylight saving time** to estimate NYC opinion, treating opinion **convergence** (loss of realistic diversity) as a layered, cross-cutting risk rather than an end-of-project patch.

## Routing — read in this order
- `docs/conventions.md` — house style, the single source of truth for *how* we work. **Read first, every session.**
- `docs/status.md` — current phase, next unit, open threads. **Read at session start.**
- `docs/design/polis-object.md` — what the object is + its interface. `docs/design_layers.md` — the **R-number architecture rules** (the ADR template's "ARCHITECTURE.md"). NB: only R1–R17 are defined there; the phase plan also cites R18–R21 (not yet written up).
- `docs/implementation_rough_plan.md` — **the authoritative phase plan**: 8 phases (walking skeleton → full survey), each with spike question(s), DoD, layers touched, rules activated.
- `docs/adr/` — ADRs: why a choice was made, options weighed, rules touched (`docs/handoff_template.md` is the ADR format). Don't re-litigate.
- `docs/briefs/` — the current unit's brief. `docs/checkpoints/` — phase-gate notes.
- Background: `docs/project_brief.md` (direction) · `docs/requirements_features.md` (subsystems under the prompt).

## The object
`src/polis/` is the deliverable — a source-first opinion-simulation engine. Headline ops: `Population.from_census(...)` → seed; `population.survey(question)` → answer distribution; `Simulation(pop, topology).run(ticks)` → per-tick logged trajectory + divergence metric. Notebooks import *down* from `src/`; they never redefine its logic. Orchestration is via **LangGraph** (agent fan-out/gather); structured output via vLLM grammar-constrained decoding.

## Inputs / data sources
- **LLM endpoint (persona backend):** local **vLLM** serving `Qwen/Qwen3-8B-AWQ` (AWQ), OpenAI-compatible at `http://localhost:8000/v1`. GPU: RTX 4070Ti SUPER (16 GB), otherwise free. Model identity/version is pinned & logged per run (R6). See ADR 0001. Launch:
  ```bash
  vllm serve Qwen/Qwen3-8B-AWQ --quantization awq \
    --gpu-memory-utilization 0.85 --max-model-len 32768 \
    --host 0.0.0.0 --port 8000
  ```
- **Persona seeds (census):** NYC demographics — occupation, voting-age band, sex, race, industry, education (+ family circumstances if available). *Dataset TBD — to confirm (likely ACS PUMS for NYC PUMAs).*
- **Opinion seeds (optional, P2+):** DST posts scraped from X. Optional/deferred — ToS-sensitive.
- **Validation ground truth:** a published DST opinion figure for NYC/US to calibrate against. *Source TBD — to confirm.*

## Environment
- Python **3.12** via the `py` launcher / `.python-version`. ⚠️ current `.venv` is **3.11** — reconcile before installing deps.
- Run code with `py` (the bare `python` alias is the Windows Store stub).

## Working rules (from conventions.md)
- **Commits are the user's.** Never stage or commit; surface diffs and let the user drive. Branch per unit; never commit to `main`/`master`.
- Source-first: a notebook that needs a capability means adding it to `src/` with a test, then calling it. Version via git tags, not `_V2` filenames.
- Keep sessions scoped; end a state-changing session by updating `status.md` and writing the handoff.
