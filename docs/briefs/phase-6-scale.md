# Brief — Phase 6b: Scale to full population (N→100)

**Phase:** 6 (unit B of 2) · **Rules activated:** R5 (per-agent-per-tick scheduling at
scale), R6 (model pinned/logged per run), R17 (run versioned against config) · **Rules
reused:** R7 (measured via voice diversity), R16 (null floor + demographic-only arm),
R14/R15 (voice-cluster / within-cell dispersion readout) · **Layer:** Architecture
(re-validated) + Validation (readout reused) · **ADRs:** 0015 (USA pivot), 0016
(disposition pipeline).

## Context

P6a builds the persona-realism pipeline → a versioned **persona corpus** artifact. P6b
is the **scale-out**: push N to 100, re-run the P3 infra benchmark, and run the 3-arm
voice-diversity contrast — all consuming P6a's corpus via `Population.from_corpus`.
Split from P6a so one unknown lands per gate: P6a proves the personas, P6b proves the
infra. The phase-plan DoD is deliberately blunt — *a full 100-agent run completes with
clean logs* — because "nothing new architecturally should be discovered here if 0–5
were done properly; this is where that assumption gets tested."

Population is the **USA** (ADR 0015); validation against a published US DST poll is
**deferred to P7** (R21). P6b produces the estimate + the spike readout, not the
calibration.

## The 3-arm run (ADR 0016)

A ladder of anchoring over the **same** 100 ACS draws, on `small_world`,
`qwen/qwen3-32b`, model pinned/logged (R6), full config versioned + corpus cited (R17):

1. **null** — one generic persona ×100 (P5 `NULL_PERSONA`): the collapse floor.
2. **demographic-only** — factual row description, empty anchors: the **stereotype
   arm** (ADR 0016's rejected estimator, kept as an instrument).
3. **full-pipeline** — P6a corpus (real-microdata disposition anchor + seeded
   backstory/facts memories).

`null→demographic` isolates demographic differentiation; `demographic→full-pipeline`
**directly tests the stereotype thesis** — does demo-only show collapsed within-cell
variance / exaggerated between-cell separation vs. the real-disposition arm? Readout is
the 5B voice-cluster count + pairwise dispersion (threshold swept at N=100), plus a
within-cell dispersion measure; R27 action-space adequacy read *first*.

## Survey-path resilience — mostly already done; close the observability gap

Correction to the P5 debt note: per-agent **retry+skip already exists** (`graph._ask` →
`retry_on_llm_error`, added in `phase-5-hardening`/`9bc2985`, merged to `main` — *after*
the P5 checkpoint that flagged it missing). The Qwen3 whitespace blob surfaces as an
`LLMError` in `choose` and is retried then skipped. **The only real gap at N=100:**
skips are **silent** — `_ask` drops a failed agent with no count/log, so a survey that
loses k agents looks like a clean (100−k) result. Close that: surface a **skipped count
+ ids** from `run_survey` so a degraded survey is visible, not invisible.

## Definition of done

- **`run_survey` hardened**: per-agent retry then skip; one flaky agent cannot abort a
  100-agent survey; returns survivors + a skipped count.
- The **3-arm run at N=100 × 5 ticks** completes with **clean logs** (0 decide
  failures, or failures surfaced — not hidden); model pinned/logged (R6); config
  versioned + corpus cited (R17).
- **Infra re-validated at scale (R5/R6):** the P3 throughput aggregate
  (`decides_per_s`, latency p95, tokens, est cost) logged at N=100 and compared to the
  P3 N=25 baseline — does the infra plan hold, or does something non-linear break?
- **Spike answered:** per-arm voice-cluster count + dispersion (+ within-cell
  dispersion) reported; does the full-pipeline arm show more realistic diversity than
  the demographic-only (stereotype) arm? R27 gate read first.
- Deterministic suite green (P6a's + the `run_survey`-harden tests); P0–P5 unchanged.

## Prerequisites

P6a complete: `Population.from_corpus` + a committed persona-corpus artifact. Branch
`phase-6b-scale`. **External:** the live run waits on the real corpus **and** explicit
spend approval (3 arms × 100 agents × 5 ticks ≈ 1500 decides + 3 surveys of real
OpenRouter spend). Deterministic work (the `run_survey` harden) proceeds now.

## Ordered tasks

1. `src/polis/graph.py` — retry+skip is done; **add skip observability**: have
   `run_survey` return survivors **plus** the skipped agent ids/count (and log them),
   so a degraded survey is visible. TDD `tests/test_graph.py` (fake client: transient
   failure → succeeds on retry; one permanently-failing agent → skipped, survey still
   returns the rest *and reports the skip*).
2. Metrics: confirm `cluster_count` / `pairwise_dispersion` accept a swept threshold
   list at N=100; add a within-cell dispersion helper if the stereotype readout needs
   it. Reuse the 5B readout otherwise.
3. `notebooks/experiments/phase6_scale.ipynb` — the live 3-arm run + throughput
   compare + voice-cluster/within-cell readout + R27 gate + verdict. (Runs on corpus +
   spend approval.)
4. `docs/checkpoints/phase-6.md`; flip ADRs 0015/0016 to accepted; enact the held
   USA-pivot doc updates (`CLAUDE.md`, `PHASE_PLAN.md`, README, project brief).

## Acceptance checks

- `.venv/Scripts/python.exe -m pytest -q` green.
- `test_graph.py`: a survey with one permanently-failing agent returns the other
  answers + skipped count 1 (no raise); a transiently-failing agent succeeds on retry.
- (Live, on corpus + approval) 3-arm N=100 run completes; `Run.throughput` logged and
  compared to P3; per-arm voice-cluster + dispersion + R27 gate recorded to
  `data/phase6_*.csv`.

## Hand-off pointer

Update `status.md` on the live-run handoff: the spike answer (does the real-disposition
arm beat the stereotype arm on realistic diversity?) and whether the infra plan held at
100 agents. P7 inherits: the matured `run_survey`, the USA population as the calibration
frame, and the owed US-DST-poll ground truth (R21).
