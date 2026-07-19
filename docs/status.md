# Status

**Phase:** pre-P0 — scoping & scaffold (wrapping up)
**Updated:** 2026-07-19

## Now
Scaffold laid down: project tree (per conventions), README phase map (mirrors `implementation_rough_plan.md`), `CLAUDE.md`, `docs/design/polis-object.md`, ADR 0001 (LLM endpoint), and the Phase 0 brief. Object agreed = `polis`. Phase plan is authoritative in `implementation_rough_plan.md` (8 phases, walking skeleton → survey maturity). Awaiting review + your commit.

## Next unit
Kick off **Phase 0 — walking skeleton** (see `docs/briefs/phase-0.md`). First task: reconcile the Python env, install deps (LangGraph + OpenAI-compatible client + pydantic + pytest), stand up the vLLM server, and get 3 hardcoded agents to return schema-valid JSON to one survey question.

## Open threads
- **Push:** no `origin` remote set yet — need the GitHub URL (`gh` not installed). Commits are yours.
- **Env:** `.venv` is Python 3.11 but the pin / `py` is 3.12 — reconcile before installing deps.
- **Endpoint:** persona model is served by **remote** vLLM (RunPod/Koyeb), OpenAI-compatible — host is ephemeral, set per session via `POLIS_LLM_BASE_URL`. Call details + params: `docs/query_handbook.md`. Local 4070Ti reserved for lightweight local models (e.g. Phase 1 embeddings).
- **Reference:** Concordia cloned into `reference/reference/` (carries its own `.git`) — gitignored so it doesn't create a broken embedded gitlink; kept locally for reading.
- **Undefined rules:** the phase plan cites **R18–R21**, but `design_layers.md` defines only R1–R17. Need writing up (which rules are these?).
- **Doc reconciliations:** (a) `conventions.md` describes date-named `decisions/`; plan + `handoff_template.md` use numbered **ADRs in `docs/adr/`** — going with `docs/adr/`; reconcile `conventions.md` wording. (b) ADR template cites `ARCHITECTURE.md`; the R-rules live in `docs/design_layers.md` — same doc, or rename? (c) `handoff_template.md` holds an ADR (decision) template, not a session-handoff template — want a separate handoff template too?
- **Census dataset:** which source for NYC demographic seeds? (likely ACS PUMS) — to confirm (Phase 5 seeding, but decide early).
- **Validation ground truth:** which published DST opinion figure do we calibrate against? — to confirm (Phase 5/7).
- **viz_theme:** categoricals deferred until the persona schema is fixed (~Phase 5).

## Handoff format
On a state-changing session end: update this file — what changed, what's live vs half-done, decisions + why, blockers. Any decision gets an ADR in `docs/adr/` (format: `docs/handoff_template.md`).
