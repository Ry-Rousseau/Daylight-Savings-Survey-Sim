# Checkpoint — Phase 6: Scale to N=100 + the realism struggle

**Status:** Phase-6 DoD met (full 100-agent runs complete with clean logs; R5/R6/R17
re-validated at scale). But the phase's *real* story is the one the DoD doesn't name —
**can a census-seeded silicon sample produce a *realistic* opinion distribution?** —
and that is only *partially* resolved. This checkpoint documents the struggle in detail
because it is the pickup point for the next sessions.

---

## The central challenge: getting a realistic result

Read this section first. The technical deliverables (below) are in service of it.

### 1. Persona realism — demographics alone are a stereotype amplifier (ADR 0016)
We first planned a **demographics-only** estimator (factual description, empty anchors)
and reversed it: given only demographic labels, the model infers disposition from its
*conditional mode* — the stereotype centroid — which collapses within-cell variance and
exaggerates between-cell separation. Fix: inject **real disposition variance** from donor
surveys (ANES TIPI/values, GSS trust/hours/happiness, ATUS wake time) via Gower 1-NN
matching, then have the LLM only *elaborate* fixed real inputs (backstory + a
reflection→anchor). This built a defensible population — but did **not** by itself buy a
realistic *stance* distribution (see below).

### 2. The collapse — a rich population still flattened onto a model prior
The first full 100-agent run (Qwen3-32B, small-world, 5 ticks) collapsed to **100%
permanent standard time on both the vote AND the voice** (1 voice cluster). Crucially,
**tick 0 was already ~92% standard, before any interaction** — so this was an
*individual-decision* failure, not a dynamics one. The R29 retrieval provenance showed
the model *was* using the persona (wake-time memories surfaced) but applying a **shallow
one-feature heuristic** (early riser → morning light → standard) and **flattening a
genuinely diverse input** (wake times span 4am–3pm, real night owls present) onto a
uniform answer.

### 3. The prior is *expert consensus*, not public opinion (the key insight)
Calibrating against the **YouGov 2023 DST survey** (the ground-truth instrument we
adopted this phase) exposed the mechanism. On the two questions:
- **Q2 (eliminate the clock-change?)** — models calibrate *well* (qwen-max+reasoning
  hit 68/21/11 vs real 62/21/17). Here **expert and public agree**.
- **Q4 (which time permanent?)** — models fail *catastrophically and robustly*: every
  Qwen model says ~85–99% **standard**; reality is **50% DST / 31% standard**. Here
  **expert and public disagree** — permanent standard time is the sleep-science/expert
  consensus, permanent DST is the lay preference (long summer evenings).

**Finding: the LLM expresses the normative/expert answer and overrides even
demographically-grounded personas wherever public opinion diverges from expert opinion.**
This was **robust to model size (32B→qwen-max) and to reasoning** — neither moved Q4 off
the standard-time prior; reasoning actually pushed toward *more* "keep switching."

### 4. The model swap — the individual collapse was largely a Qwen artifact
Switching the decide/survey model to **Claude Sonnet-5** broke the anti-DST prior:
YouGov Q4 TVD 0.22 (vs Qwen 0.52–0.68), **45% DST** individually (vs Qwen 0–12%), and it
held **~83 distinct voice clusters** among 100 agents at scale. So the individual-level
collapse was a *model* artifact, not a failure of the persona pipeline. **This is the
headline result and the strongest argument of the project.** (A one-arm family test —
Sonnet — was enough to answer "is the prior universal?"; DeepSeek/GPT/Gemini remain
untested but were deprioritized once Sonnet worked. Gemini-3.1-pro *does* work but has
mandatory reasoning + JSON-truncation friction.)

### 5. Residual convergence is *genuine dynamics*, not the model
Even Sonnet's diverse start (tick-0 dominant 0.66) still converges the *vote* under
5-tick small-world contagion (→0.92), while the *voice* stays diverse. So there are **two
independent convergence sources**: the model-collapse artifact (fixed by a better model)
and real R10 interaction dynamics (the phenomenon the project exists to study). We
dissected the latter: topology alone doesn't help (ring ≈ small-world, because the
standard lean is uniform); only a **committed minority** preserves a split (0.56);
removing vote-broadcasting (**deliberate** discourse mode) helps partially (0.98→0.86)
but reasons carry the population's bias too.

### 6. Opinion seeding is too weak to steer — and the X corpus is noisy
The "attractor-flip" test (conviction-seed a DST plurality, amplify on a dense graph)
**failed**: 40% DST-seeded → only 14 DST at tick 0. Cause: the X conviction corpus is
**news-heavy and partly mislabeled** ("*this is how I honestly feel: the House passed a
bill…*"; one tweet labeled pro-DST literally says "end daylight savings time"). Sonnet's
persona reasoning overrode the weak/incoherent seeds. To actually steer the vote you need
**committed seeding** (mechanically forces it) or a **cleaned opinion pool** (drop
headlines/URLs/mislabels).

### 7. The demographic *structure* is mis-calibrated (newest, half-analyzed)
Even with Sonnet's roughly-right *aggregate* (~37–45% DST vs YouGov 50%), the
**demographic gradient is wrong**. Per-persona analysis (Cramér's V, binned to comparable
cardinality) found **no demographic strongly predicts stance (all < 0.2)**; age is the
cleanest but **inverted vs reality**:
- Ours: younger → more DST (18-34 ≈ 52% → 65+ ≈ 26%).
- YouGov Q4 crosstab (pp. 7–8): flat/rising with age (18-29 = 45, 45-64 = 54, 65+ = 52).

And the *lifestyle* predictors one would expect (chronotype/wake-time 0.14, employment
0.00) are near the bottom — Sonnet forms the opinion from broad demographics, not a
"night-owl → evening light → DST" chain. **This is the live pickup thread** (see below).

### 8. Question wording swings the result hugely (watch this)
Sonnet on the **DST_QUESTION** (4 options incl. "keep switching") gives ~2% DST at tick 0;
on **YouGov Q4** (permanent DST vs standard) gives ~37–45%. Keep the two straight: the
**YouGov figure is the "matches reality" headline**; the **DST_QUESTION tick-trajectory is
the "watch it converge" dynamics story**. They are not directly comparable.

---

## What was built (deliverables, all committed to `main`)

- **Persona pipeline (P6a):** `persona_pipeline.py` (`SeededPersona`, `build_corpus`,
  `Population.from_corpus`, corpus JSON + R29 provenance, cross-donor sanitize +
  contradiction flags, stage-differentiated seed models). Corpus:
  `data/personas_corpus_n100.json` (gitignored; regenerate via
  `notebooks/engineering/persona_pipeline.qmd` / `sandbox/run_persona_corpus.py`).
- **Opinion/conviction layer:** `opinion_seeds.py` (X-corpus seeding, two-camp/random,
  committed minorities R11).
- **Action space (Parts A+B):** `SHARE_CONSIDERATION` (reason, no tally, ADR 0017) +
  `REBUT` (pushback that tallies, ADR 0018) + `deliberate` discourse mode
  (`DynamicsConfig.discourse_mode`). 176 tests green.
- **Calibration instrument:** `questions.py` `YOUGOV_Q2/Q4` + toplines as ground truth.
- **LLM robustness:** the `json` keyword + tolerant key/value extraction
  (`match_option`/`extract_choice`) for cross-provider structured output; skip
  observability; scheduler `on_progress` + `Simulation.run(on_tick=…)`.
- **Visualization:** `viz_theme.py` (validated Okabe-Ito stance palette) +
  `sandbox/build_figures.py` + `sandbox/build_sonnet_trajectory.py` +
  `sandbox/analyze_predictors.py`; figures in `output/figures/` (tracked); report
  `notebooks/experiments/phase6_report.qmd`.

ADRs this phase: **0015** (NYC→USA pivot), **0016** (disposition pipeline), **0017**
(SHARE_CONSIDERATION), **0018** (REBUT). A model-swap ADR (Qwen→Sonnet for calibration
realism, amending 0005) is **owed** — capture the §3–4 finding.

---

## Pickup points for the next session (ordered)

1. **Finish the our-vs-YouGov demographic comparison** (in flight, uncommitted):
   `sandbox/build_comparison.py` is written but **not yet run/committed**. It plots
   %-DST by age/sex/region, ours vs YouGov. The YouGov Q4 crosstab is transcribed there
   (age 45/43/54/52; sex M45/F54; region NE53/MW52/S49/W45). Run it, verify, commit. The
   **inverted age gradient (§7)** is the finding to foreground.
2. **Fix the demographic mis-calibration (§7)** — this is the deepest open realism
   problem. Options to explore: does the *inverted age gradient* come from Sonnet's own
   prior, or from the persona *content* (are our older personas' disposition seeds
   pushing them to standard)? Probe with the R29 provenance on old-vs-young personas.
3. **Clean the opinion corpus (§6)** and re-run the attractor-flip with **committed**
   seeding — the "can we steer the vote / model factions" thread.
4. **Model-swap ADR** (owed) + optionally a cheap DeepSeek/GPT Q4 check to confirm the
   anti-DST prior is universal, not Qwen-specific (Sonnet already shows it's *breakable*).
5. **Adopt Sonnet as the decide/survey baseline** in config/docs (currently Qwen per
   ADR 0005; the whole realism result depends on Sonnet).

---

## Known issues / gotchas (save future-you the pain)

- **`build_figures.py` has mojibake** — a re-save corrupted the en/em-dashes in several
  subtitles (`50â€"31`). The *committed PNGs are fine*; **re-running the script will
  regenerate garbled subtitles**. Fix the dashes before regenerating.
- **Per-option calibration data is a reconstructed CSV.** The calibration runner saved
  only TVD; the per-option %s live in `data/phase6_calibration_full.csv`, hand-entered
  from the run logs (the Sonnet run overwrote the Qwen progress log). Numbers are real
  but not machine-regenerable without re-running the calibration.
- **Provider quirks (now handled in `llm.py`):** DashScope's Qwen3-235B needs the word
  "json" in the messages for `response_format`; several providers ignore the schema's
  *key names* (return `{"response":…}` not `{"choice":…}`) — `extract_choice` tolerates
  both. Gemini-3.1-pro has *mandatory* reasoning (can't disable) + truncates JSON under a
  small `max_tokens`.
- **Worktree subagents branch from the last *commit*, not the working tree** — a subagent
  spawned mid-session built on a stale base and needed a 3-way merge. Commit before
  spawning worktree agents.
- **`data/` and `reference/` are gitignored** (regenerable / vendored). Result CSVs +
  run logs + the corpus live there and are NOT in git; the figures in `output/figures/`
  ARE tracked and are the durable deliverable.
