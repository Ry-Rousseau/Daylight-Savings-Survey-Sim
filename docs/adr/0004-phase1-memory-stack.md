# ADR 0004: Phase 1 memory stack

Status: accepted
Phase: 1 — Memory

## Spike question

Which local embedding model is fast enough at this scale? Does
recency/importance/relevance retrieval actually change output — and what is
the simplest store that keeps per-agent memory structurally private (R2)?

## Context

Phase 1 thickens exactly one axis — memory — over the Phase 0 walking
skeleton. The DoD: two agents with different seeded memories give measurably
different survey answers. That requires (a) an embedding backend, (b) a
per-agent vector store, (c) a memory record schema, (d) a retrieval scorer,
and (e) an integration point that reads memory *before* the unchanged
`LLMClient.choose()` contract and writes survey answers back (R19).

Constraints in play:
- **R2** — global/environment state and per-agent private state (memory,
  beliefs) must be separated *structurally, not by convention*.
- **R19** — survey responses are written back into the responding agent's
  memory stream as an event, so repeated surveys stay coherent.
- **Scale is tiny at P1** — ~2 demo agents; ~100 at P6 × ~20 memories ≈ 2k
  vectors. Every "which is fast enough / which DB scales" question is moot at
  this scale; brute-force cosine over a few thousand 384-dim vectors is
  sub-millisecond. So the memory decisions optimize for R2 cleanliness and
  simplicity, not throughput. Throughput is a P3/P6 concern.
- **Env** — RTX 4070 Ti SUPER (16 GB) present; `torch`/`numpy`/embedding libs
  were *not* installed (greenfield deps), so dependency weight is a live cost,
  not sunk. OpenRouter is chat-only — embeddings are a separate, local path.

## Options considered

**Embedding backend**
1. **sentence-transformers + BGE-small on GPU** — CUDA torch (~2.5 GB install),
   `BAAI/bge-small-en-v1.5` (384-dim). Matches the CLAUDE.md direction
   ("reserve the 4070Ti for local embeddings"); pays off at P6 scale.
2. fastembed (ONNX, no torch) — lighter dep tree, CPU-fast at this scale, but
   deviates from the documented "GPU for embeddings" plan.
3. Hosted embeddings API — no local install, but a second paid endpoint to
   manage and a net-new external dependency.

**Vector store**
1. **Per-agent in-memory numpy store, brute-force cosine** — each `Agent` owns
   its own store instance; there is structurally no shared index to leak.
2. One global Chroma / sqlite-vec keyed by `agent_id` metadata filter —
   enforces R2 *by convention* (a filter you can forget); the R2 anti-pattern.

**Importance source**
1. **Pluggable `importance_fn`** — seeds carry authored importance (1–10) by
   default; an LLM poignancy rater is selectable via config.
2. LLM poignancy call on every memory — Park's approach; per-memory cost even
   for authored seeds.
3. Heuristic only — no LLM path at all.

## Decision

- **Embeddings:** sentence-transformers `BAAI/bge-small-en-v1.5` on GPU
  (CUDA torch). Wrapped behind `src/polis/embeddings.py` so the model is a
  swappable field, not a global.
- **Store:** per-agent in-memory numpy `MemoryStore` (brute-force cosine),
  JSON-serializable for seeding/save. No external vector DB at P1.
- **Record schema:** `MemoryRecord{ text, created_at, last_accessed_at,
  importance (1–10), embedding, kind }`. `kind` ∈ {seed, survey, …} keeps
  reflection records forward-compatible with no migration.
- **Importance:** pluggable `importance_fn`; authored on seeds by default, LLM
  poignancy rater selectable via config.
- **Retrieval scoring:** Park et al. defaults — `score = w_rec·recency +
  w_imp·importance + w_rel·relevance`, each min-max normalized across the
  agent's memories, weights all `1.0`, decay `0.995`, `top_n=5`. All fields
  live on a `RetrievalConfig` dataclass — nothing hardcoded.
- **Recency time axis:** `created_at` authored on seeds at P1 (no sim clock
  until P2's tick loop); the real clock replaces it at P2.
- **Prompts:** a single `src/polis/prompts.py` holds every stage's prompt
  (persona system, memory-injection block, survey user prompt, poignancy
  rating) so the homogenization-relevant surface is auditable and ablatable.
- **Reflection:** deferred. P1 is retrieval-only.
- **DoD metric:** distribution over K=20 samples per agent; metric =
  P(permanent-DST) delta, with an empty-memory control proving memory is the
  cause. Demo lives in an experiments notebook; the pytest stays
  deterministic (scoring math, no network).

## Why

- **R2 is the load-bearing rule this phase**, and a per-agent numpy store makes
  it *structurally* true: there is no shared index, so cross-agent leakage is
  impossible by construction rather than prevented by a metadata filter that a
  future edit could drop. A global vector DB would trade this away for scale we
  do not need until P6.
- **Scale makes the embedding-speed question moot**, so the deciding factor is
  the documented direction (GPU-local embeddings, chat endpoint stays paid-only)
  and P6 headroom — hence GPU BGE-small over a hosted API.
- **Authored importance avoids per-memory LLM cost** for seeds we write
  ourselves, while the pluggable `importance_fn` keeps the LLM poignancy path
  available where it earns its cost later.
- **A P(permanent-DST) delta over 20 samples is honest about temperature
  noise** in a way a single flipped choice is not; the empty-memory control is a
  mini R16 null-model that attributes the difference to memory, not persona.

## Consequences

- Locks in `torch` + `sentence-transformers` + `numpy` as P1 dependencies
  (added to `pyproject.toml`). Revisit the store choice **if agent count ×
  memories exceeds what brute-force cosine handles comfortably** (expected
  ~P6); at that point a per-agent-namespaced DB, not a global filtered one.
- `LLMClient.choose()` stays unchanged; memory is additive inside
  `Agent.answer()`. R19 writeback is implemented here.
- Recency depends on an authored `created_at` at P1 — a scaffold, not the final
  mechanism; the P2 sim clock supersedes it.
- Reflection remains unbuilt; the `kind` field reserves space for it.
- Persona is held constant for the DoD test (isolates the memory axis);
  Persona-layer depth proper is P5.

## Rules touched

- **R2** — implemented: per-agent numpy store enforces private-state separation
  structurally.
- **R19** — implemented: survey answers written back to the agent's memory
  stream as events.
- **R1** — respected: embedding model and retrieval config are fields, not
  globals.
- **R6** — unchanged: chat model still pinned/logged per call; embedding model
  identity logged alongside.
