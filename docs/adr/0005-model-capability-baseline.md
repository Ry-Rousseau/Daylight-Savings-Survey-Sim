# ADR 0005: Model-capability baseline — Qwen3-32B

Status: accepted
Phase: 1 (addendum)

## Spike question

The Phase 1 DoD passed at Qwen3-8B only after the DST survey options were
annotated with their sunrise/sunset consequence. Is that a weak-model artifact?
Does the memory effect survive at larger models, is the annotation a crutch only
the 8B needs, and does model size change baseline convergence — i.e. does ADR
0002's 8B baseline hold?

## Context

Phase 1 showed 8B *reasons* correctly from seeded memory ("an 8am winter sunrise
horrifies me") but couldn't map "permanent daylight saving time" → later
sunrises, so a memory-driven preference landed on the wrong option until the
options were annotated. R6 states model identity carries different baseline
convergence tendencies, so model capability deserves to be a *measured* axis, not
an assumption. We are currently on managed multi-model OpenRouter (ADR 0002) —
the cheap window to test this before P5 self-hosting pins a model. `LLMConfig.model`
is already a per-agent field (R1), so the sweep is a config change, not new code.

Ladder chosen: same-family **Qwen3 dense 8b → 14b → 32b** (verified against the
live OpenRouter model list — there is *no* Qwen3-72B; the 72B was Qwen2.5, a
different generation). Same family isolates size from lineage; all three are
self-hostable, keeping ADR 0002's self-host plan intact. Grid: 3 models ×
{annotated, plain} wording × {evening, morning, empty} × K=15 = 270 samples.
Runner: `notebooks/experiments/phase1_model_sweep.ipynb` + `data/phase1_model_sweep.csv`
(records the chosen option *index* so wordings are comparable).

## Findings

- **Memory delta survives at every size (annotated):** P(permDST|evening) −
  P(permDST|morning) = **1.0 at 8b, 14b, 32b**. Retrieval-driven answer control
  is not a small-model artifact.
- **The annotation is an 8B-only crutch:** P(permStandard | morning) under *plain*
  wording = 8b **0.00**, 14b **1.00**, 32b **1.00**. Without annotation the 8B's
  memory effect collapses entirely (plain delta 8b **0.0** vs 14b/32b **1.0**);
  14B+ carry the "permanent DST = later sunrises" world-knowledge unaided.
- **Baseline convergence (weak proxy):** empty-control normalized entropy = 8b
  **0.0**, 14b **0.0** (both collapse to one answer), 32b **~0.42–0.49** (spread).
  The larger model is *less* collapsed at baseline — opposite of the naive
  "bigger = more homogenized" guess. Caveat: single-persona sampling spread on one
  question, not multi-agent convergence (the real test needs diverse personas +
  interaction, P4/P5).

## Options considered

1. **Keep Qwen3-8B + always annotate options** — cheapest/fastest at 100-agent
   scale, but every survey question needs consequence-annotated options
   (researcher framing per question) and the memory effect is fragile to wording.
2. **Revise to Qwen3-14B** — removes the crutch, full plain-wording delta,
   cheapest model that does so, comfortably self-hostable. No measured downside.
3. **Revise to Qwen3-32B** — also removes the crutch; adds capability headroom
   and (on this probe) more baseline sampling spread; ~4× 8B cost; borderline
   self-host on a single 4070Ti (fine on the P5 RunPod plan).

## Decision

**Option 3 — baseline is `qwen/qwen3-32b`** (`llm.py:DEFAULT_MODEL`). Amends the
model pin in ADR 0002; the provider phasing (OpenRouter P0–2 → self-hosted vLLM
P5+) is unchanged. P5 self-host target becomes `Qwen/Qwen3-32B-AWQ` (~20 GB,
within a 24 GB RunPod GPU).

## Why

Chosen for capability headroom within the family and to remove any dependence on
a wording crutch, at the cost of ~4× 8B token cost. (Recorded for future-you:
14B met the same robustness bar more cheaply and was the analysis's recommended
option; 32B was chosen deliberately for headroom. If P6 100-agent cost/latency
bites, 14B is the drop-back that still clears every check here.)

## Consequences

- The Phase 1 DoD is confirmed at the new baseline: 32B annotated delta = 1.0,
  and unlike 8B it also holds under *plain* wording (delta 1.0) — the P1 result
  is stronger, not weaker, at 32B.
- Live tests and notebooks now hit 32B (slower, ~4× cost). The historical P1 DoD
  evidence remains recorded at 8B in `checkpoints/phase-1.md` — that run is not
  re-executed.
- `DST_QUESTION` stays consequence-annotated (canonical). With 32B the *plain*
  variant (`DST_QUESTION_PLAIN`) is now equally viable; whether to prefer plain
  wording for opinion-estimate validity (less framing) vs annotated (matches
  real-poll practice) is deferred to the validation phase (P5/P7).
- Revisit if P6 100-agent cost/latency is prohibitive → fall back to Qwen3-14B.

## Rules touched

R1 (per-agent model field), R6 (model identity/convergence tendency — the reason
this sweep exists), R16 (null-model baseline spirit — the empty control).
