# Project Conventions — hybrid data-science / engineering

Single house-style doc for a project that **builds and iterates an object** (model / parser / scorer / service / pipeline / agent) with a **research surface** (EDA + experiments) beside it. Read before touching the repo. Where a project-specific value is missing, ask or infer-then-confirm — never hard-code a guess.

## Shape

The **object is the deliverable**; notebooks import it to probe / EDA / eval it, never the reverse. Work runs in **planned phases with checkpoints**, each a short arc of scoped units. Scope up front, execute inline, write down why.

    ProjectName/
    ├── src/<package>/   # the object — source-first, tested (the trunk)
    ├── tests/           # first-class pytest — the net for every iteration
    ├── notebooks/       # research surface: engineering/ · eda/ · experiments/
    ├── evals/           # repeatable quantitative eval (harness + sets + results), if the loop recurs
    ├── data/            # build: generated data / fixtures; starts empty
    ├── output/figures/  # build: figures (stable names)
    ├── docs/            # conventions.md · design/ · decisions/ · briefs/ · checkpoints/ · status.md
    ├── reference/  ·  sandbox/  ·  CLAUDE.md  ·  README.md

`src/ tests/ notebooks/ docs/` are authored; `data/ output/` are regenerable build artifacts — never overwrite or delete a generated file without confirmation. On a small project, fold `briefs/ checkpoints/` and handoffs into `status.md` + `decisions/`.

## Notebooks

- **One notebook = one question**; the filename says which. Split when it grows a second.
- Anatomy, in order: header (the aim, 1–3 lines) → imports → **one `ALL_CAPS` config cell** ("edit and re-run downward") → data load with sanity prints → §-numbered sections with **narrative above the figure**. Markdown is flowing prose (one paragraph per source line, no hard-wrap).
- **Figures are plotnine** (matplotlib only where plotnine can't go, e.g. NetworkX, themed to match). Colors/labels from a `viz_theme` module, never raw hex; explicit sizes and axis limits.
- Notebooks **import the object from `src/`**; they never redefine its logic. An `experiments/` notebook = one hypothesis about the object, run against a fixed set, ending in a verdict that feeds a `decisions/` note.

## Code in `src/`

Default to inline notebook code; promote to `src/` when **reused (2+ notebooks), expensive/external, stateful/risky, or a domain primitive** (named after the concept). Three similar lines beat a premature abstraction.

**Engineering inverts this.** When the object *is* the deliverable it's source-first in `src/` from day one — tested, behind a narrow interface, with notebooks importing *down* from it. Code doesn't graduate up out of notebooks; a notebook that needs a capability means you add it to `src/` with a test, then call it. Version the object through git tags, not `_V2` filenames.

## Workflow — phases, checkpoints, units

- **Phases:** plan the arc up front — numbered phases, each a coherent capability step; map in `README.md`, position in `status.md`. Re-plan deliberately (a `decisions/` note), not by drift.
- **Checkpoints:** each phase ends with a short `checkpoints/<phase>.md` — what it proved, what's live, what the next phase needs. Document at gates, not per-unit and not only at the end.
- **Units:** a phase is a few scoped build units, each bookended by a **brief** (before — context · definition-of-done · prerequisites · ordered tasks · acceptance checks · hand-off pointer) and a **handoff** (after, mirrored to `status.md` — what changed, what's live vs half-done, decisions + why, blockers).
- **Spikes are the exception** — only for a genuine open question that blocks a build; output a short `decisions/` doc, then resume. Most work goes brief → build → handoff.
- **Execute inline:** Plan Mode (explore → plan → approval → build in the same session), TDD where it fits. Don't railroad with heavy plan/subagent skills — they over-prescribe and drop repo conventions. **Commits need explicit approval- I like when I control the commits since I like reviewing changes before they are staged**

## Code quality (every unit)

Match the surrounding code (patterns, naming, error-handling). Extend a module over adding a file. Comments give vital context — what / how / why, for the next reader — not narration; keep phase/migration/philosophy backstory out of source (it lives in the brief, the decision note).

## Docs & session loop

- `design/` = what the object is · `adr/NNNN-*.md` = numbered ADRs (why a choice was made, options weighed, rules touched — a rejected-options ledger; don't re-litigate), format in `adr/template.md` · `status.md` = current phase, next unit, open threads.
- **Session start:** read `conventions.md`, `status.md`, the current brief. **Session end (state changed):** update `status.md`, write the handoff, record any decision.
- `sandbox/` = free scratch, deletable; anything worth keeping graduates out.
