# Checkpoint — Phase 1: Memory

**Date:** 2026-07-20 · **Status:** ✅ complete (DoD met) · **Branch:** `phase-1-memory`

## What it proved

Seeded memory, retrieved before `choose()`, **measurably determines** an agent's
survey answer. Same persona, opposite seeded memories, K=20 samples each on the
DST question (`qwen/qwen3-8b`, temp 0.8):

| condition | choice (20/20) | P(permanent-DST) |
|---|---|---|
| evening-seeds | permanent daylight saving time | 1.00 |
| morning-seeds | permanent standard time | 0.00 |
| empty-control | keep switching twice a year | 0.00 |

**P(permanent-DST) delta (evening − morning) = +1.000**; the empty control lands
100% on neither pole, so the shift is attributable to memory, not persona.
Evidence: `notebooks/experiments/phase1_memory_dod.ipynb`.

## What's live

- `src/polis/embeddings.py` — `EmbeddingModel` (BGE-small on **CUDA**, 384-dim,
  unit-normalized, lazy-loaded).
- `src/polis/memory.py` — `MemoryRecord` + per-agent numpy `MemoryStore`;
  Park-style recency/importance/relevance retrieval, each min-max normalized;
  `RetrievalConfig` (weights 1.0, decay 0.995, top_n 5). **R2** enforced
  structurally — one store per agent, no shared index.
- `src/polis/prompts.py` — single home for all stage prompts (persona system,
  memory-injection block, survey user, poignancy).
- `src/polis/importance.py` — pluggable `importance_fn`: authored on seeds +
  optional LLM poignancy rater.
- `src/polis/memory_seeds.py` — shared neutral persona + two contrasting seed
  sets (evening/morning) with authored importance and relative `created_at`.
- `src/polis/questions.py` — canonical `DST_QUESTION` (consequence-annotated
  options), shared by `__main__` and the notebook.
- `src/polis/agent.py` — retrieve → inject → **unchanged** `choose()` → **R19**
  writeback (survey answer stored as a `kind=survey` memory event; verified).
- Tests: 13 passing (5 Phase 0 + 8 new deterministic retrieval-scoring tests,
  no network). Deps added: torch (cu124), sentence-transformers, numpy, pandas,
  plotnine, nbconvert.

## Observed (sanity only — not a result)

- **`choose()` contract held** — memory is fully additive inside `Agent`; the
  Phase 0 LLM interface is byte-for-byte unchanged.
- **Retrieval works on GPU** — BGE-small loads on cuda; seed retrieval orders
  sensibly by the combined score.

## The finding that shaped the phase

The mechanism worked on the first live run, but exposed a real ceiling: 8B has a
saturated permanent-DST prior **and doesn't know "permanent DST = later
sunrises."** Its memory-driven *reasoning* was correct (the morning agent said
"an 8am winter sunrise horrifies me") but it mapped that onto the wrong option.
Annotating each option with its sunrise/sunset consequence — a documented
real-world DST-polling wording effect, not a thumb on the scale (the empty
control then lands on neither pole) — unlocked the clean +1.000 delta. This is
the reasoning-vs-world-knowledge distinction, and it motivates the next unit.

## What P2 (Game Master / interaction) needs

- A world-state store **separate** from agent memory (R2 again, now across the
  world/agent boundary), and a symbolic action-resolution layer.
- The abstract memory time axis (`created_at`/`now`) is a P1 scaffold — P2's
  tick loop supplies the real sim clock that recency decay should run on.

## Debt / notes

- **Deferred:** reflection (Park write-back of higher-level insights) — schema's
  `kind` field reserves space; revisit ~P5.
- **Model-capability addendum (done — ADR 0005):** same-family Qwen3
  **8b → 14b → 32b** sweep (no Qwen3-72B exists; 72B was Qwen2.5), K=15,
  annotated + plain wording. Findings: memory delta = 1.0 at all sizes
  (annotated); the option annotation is an **8B-only crutch** (plain-wording
  delta: 8b 0.0, 14b/32b 1.0); larger model *less* collapsed at baseline (weak
  proxy). **Baseline revised to `qwen/qwen3-32b`** (amends ADR 0002). Evidence:
  `notebooks/experiments/phase1_model_sweep.ipynb`, `data/phase1_model_sweep.csv`.
- tqdm `IProgress not found` warning in the notebook is cosmetic (no ipywidgets);
  console cp1252 mojibake for non-ASCII persists from Phase 0 — data is UTF-8.
- OpenRouter json_schema still soft-enforced (validate+retry); hard grammar
  (R20) returns with self-hosted vLLM at P5+ (ADR 0002).
