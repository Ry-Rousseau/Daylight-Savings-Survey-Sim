# Brief — Phase 1: Memory

**Branch:** `phase-1-memory` · **Decisions:** `docs/adr/0004-phase1-memory-stack.md`

## Context

Phase 0 gave us a walking skeleton: 3 thin personas answer one DST question
through `LLMClient.choose()`, structured output only, no memory. Phase 1
thickens exactly one axis — **memory** — and nothing else. No interaction, no
topology, no Game Master (those are P2+). Persona is held constant for the
demo so the only moving part is what an agent remembers.

The point of the phase: show that a per-agent memory store, read *before*
`choose()`, measurably shifts survey answers — and do it in a way that keeps
per-agent state structurally private (R2) and writes survey answers back into
the memory stream (R19).

## Definition of done

Two agents with the **same persona** but **different seeded memories** produce
**measurably different** answers to the DST survey question, where "measurably
different" = **P(permanent-DST) delta over K=20 samples per agent**, with an
**empty-memory control** (same persona, no seeds) showing the difference is
caused by memory, not persona. Demonstrated in an experiments notebook; the
retrieval scorer is covered by a deterministic (no-network) pytest.

## Prerequisites

- On branch `phase-1-memory` (done).
- `LLMClient.choose()` contract unchanged and green (Phase 0).
- GPU available (RTX 4070 Ti SUPER, confirmed); CUDA `torch` to be installed.

## Ordered tasks

1. **Dependencies.** Add `torch` (CUDA build), `sentence-transformers`, `numpy`
   to `pyproject.toml`; install via `uv pip install` into `.venv`. Confirm
   `torch.cuda.is_available()` is `True`.
2. **`src/polis/embeddings.py`.** An `EmbeddingModel` wrapper around
   `BAAI/bge-small-en-v1.5` (lazy-loaded, GPU, cached). Method: `encode(texts)
   -> np.ndarray` (normalized). Model id is a field, logged for provenance.
3. **`src/polis/memory.py`.** `MemoryRecord` (text, created_at,
   last_accessed_at, importance, embedding, kind) + `MemoryStore` — per-agent,
   numpy-backed, brute-force cosine. Ops: `add(record)`, `retrieve(query_emb,
   now, cfg) -> list[MemoryRecord]` (updates `last_accessed_at`), JSON
   `to_dict`/`from_dict`. `RetrievalConfig` dataclass: `w_recency=w_importance=
   w_relevance=1.0`, `decay=0.995`, `top_n=5`; each component min-max
   normalized across the agent's memories before weighting.
4. **Importance.** A pluggable `importance_fn`: authored value passed through by
   default; an optional LLM poignancy rater (prompt in `prompts.py`) selectable
   via config. Seeds carry authored importance.
5. **`src/polis/prompts.py`.** Single home for every stage prompt: persona
   system, memory-injection block ("Relevant things you remember: …"), survey
   user prompt (moved out of `agent.py`), poignancy rating. Templates, not
   scattered f-strings.
6. **Seeds — `src/polis/memory_seeds.py`** (or a `data/` fixture). Hand-written
   memory sets: one pro-permanent-DST, one pro-standard-time, each with
   authored importance and relative `created_at`. Same persona for both agents.
7. **Wire `Agent`.** `Agent` gains a `MemoryStore`. `answer(question)`: build
   query string → embed → `retrieve` top-N → inject via `prompts.py` → call the
   **unchanged** `choose()` → **R19 writeback**: store the Q+A as a `kind=survey`
   memory event.
8. **Tests — `tests/test_memory.py`.** Deterministic, no network: retrieval
   scoring (recency decay, importance weighting, relevance ranking, min-max
   normalization, top-N cutoff) with hand-built vectors. Keep existing Phase 0
   tests green.
9. **DoD notebook — `notebooks/experiments/phase1_memory_dod.ipynb`.** Same
   persona × {pro-DST seeds, pro-standard seeds, empty control}; K=20 samples
   each; compute P(permanent-DST) per condition; report the delta and the
   control. Narrative above each figure; imports down from `src/`.

## Acceptance checks

- `torch.cuda.is_available()` is `True` in `.venv`.
- `pytest` green, including new deterministic `test_memory.py`; Phase 0 tests
  unchanged and passing.
- `LLMClient.choose()` signature/behavior **unchanged** (diff shows no edits to
  its contract).
- Each `Agent` owns a private `MemoryStore` — no shared/global index (R2 grep
  check: no cross-agent state).
- Survey answers appear in the responding agent's memory as `kind=survey`
  records (R19).
- Notebook shows P(permanent-DST) delta between the two seeded agents beyond
  the empty-memory control's delta.

## Hand-off pointer

On completion: update `status.md` (what's live vs half-done, decisions,
blockers), tick Phase 1 in `PHASE_PLAN.md`, and write
`docs/checkpoints/phase-1.md` (what it proved, what's live, what P2 needs).
Commits are the user's to approve.
