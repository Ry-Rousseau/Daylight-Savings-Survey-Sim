# System Architecture

## Purpose of this document

This is the standing spec for the simulation system: what layers it's
decomposed into, why they're separated, and the first-principles rules each
layer must satisfy. It's meant to stay stable across build phases — the
`PHASE_PLAN.md` document tracks *when* each rule gets implemented; this
document tracks *what must be true* regardless of phase.

---

## First principles (why the rule set exists at all)

1. **Convergence/homogenization risk is layered, not phase-local.** It has
   independent causal mechanisms at the architecture, persona, and dynamics
   layers. A fix at one layer doesn't propagate to the others.
2. **The diversity ceiling is set architecturally, before persona work
   starts.** Whatever variance the engine structurally forecloses (shared
   state, single global temperature, hardcoded topology) can't be recovered
   later by better prompting.
3. **Unbiased components don't guarantee an unbiased system.** Strong
   collective bias can emerge in LLM agent populations even when individual
   agents exhibit no bias on their own (Ashery, Aiello & Baronchelli,
   *Science Advances*, 2025) — a clean persona layer is not evidence the
   system is safe from convergence.
4. **Every parameter you don't expose becomes a hidden assumption.** If
   temperature, topology, or memory scope is hardcoded, you can't later
   ablate it to determine how much it contributed to an observed pattern.
5. **You can't distinguish an artifact from a phenomenon without a metric
   defined in advance.** Retrofitting a divergence metric after a
   suspicious-looking run is rationalizing, not measuring.

---

## The five layers

### Layer 1 — Architecture / Engine

Everything about how the simulation runs mechanically: agent state, memory,
generation parameters, scheduling, interaction topology, and the symbolic
(non-LLM) action-resolution layer. This layer sets the ceiling on everything
above it — nothing here should be decided implicitly by convenience.

### Layer 2 — Persona

How individual agents are anchored to a distinct identity: value/disposition
content, drift monitoring, and the acknowledgment that persona diversity
alone does not prevent system-level convergence.

### Layer 3 — Dynamics

How agents interact with each other: topology, information-exchange volume,
minority/majority influence dynamics. This is the layer with the strongest
causal leverage over convergence vs. community persistence.

### Layer 4 — Validation & Metrics

Cross-cutting: how you know what happened. Metrics must be defined before a
run, not after, and logged continuously, not just at endpoint.

### Layer 5 — Interface / Query

How something *outside* the simulation queries a live population and gets
structured answers back — the survey/interview subsystem. Distinct from the
other four because it only exists when there's an external question to ask;
the population's own behavior (Layers 1–4) would need to be correct even if
no one ever surveyed it.

---

## Rule set

### Layer 1 — Architecture / Engine

| Rule | Requirement |
|---|---|
| R1 | Generation parameters (temperature, top_p, model variant) are per-agent config fields, never global constants |
| R2 | Strict data-layer separation between global/environment state (shared, read-only) and per-agent private state (memory, beliefs) — enforced structurally, not by convention |
| R3 | No shared caching or pooling of generative output across agents for efficiency reasons; any deliberately shared signal (e.g. a news feed) must be logged so its causal effect on convergence can be traced |
| R4 | Interaction topology is a pluggable, swappable parameter (fully-connected / small-world / clustered / stochastic block) — not hardcoded into the agent loop |
| R5 | Warm/cold cache scheduling (reused compute) is scoped strictly per-agent-per-tick; never merge or share cached reasoning across agents in the name of throughput |
| R6 | Model identity and version are pinned and logged per run — different base models carry different baseline convergence tendencies |
| R18 | Survey/interview is a separately invokable operation against the live population, not folded into the regular tick loop |
| R19 | Survey responses are written back into the responding agent's memory stream as an event, so repeated surveys over simulated time remain coherent rather than independently sampled |
| R20 | Survey response schema is enforced via constrained decoding, identical in kind to any in-simulation action output |
| R22 | Two distinct orchestration tools for two distinct task shapes: a custom open-ended tick loop for the simulation core, and a bounded fan-out/gather tool (LangGraph) for the query layer — one framework does not have to do both |
| R23 | Action space is a closed, versioned enumeration with a structured payload — not open-ended free text — so it stays schema-constrainable (R20) and quantitatively loggable (R14–R17) |
| R24 | Every action type has an explicit, deterministic resolution function in the Game Master before it ships — no action type is added speculatively without resolution logic |

### Layer 2 — Persona

| Rule | Requirement |
|---|---|
| R7 | Personas are anchored in values/dispositions, not just demographic labels — thin personas collapse into the model's generic "helpful assistant" voice faster |
| R8 | Persona strength is measured, not assumed — periodically probe an agent's response against its own baseline to check for identity drift over the run |
| R9 | Persona diversity is necessary but not sufficient — do not treat it as a complete fix, since collective bias can emerge from interaction structure alone with zero individual-agent cause |

### Layer 3 — Dynamics

| Rule | Requirement |
|---|---|
| R10 | Default to sparse/structured topology over full connectivity when the goal is to preserve disparate communities — full exposure biases toward global consensus |
| R11 | Build an explicit affordance for "committed minority" sub-populations if modeling faction persistence — minorities above a critical size can flip majority conventions |
| R12 | Information exchange volume per tick is a tunable parameter — consensus pressure scales with how much cross-agent exposure occurs per unit time |
| R13 | Topology must be freezable/swappable mid-run to support counterfactual comparisons (same personas, different graph) |
| R25 | `abstain`/no-op is always a valid action — forcing substantive action every tick is itself a homogenizing pressure |
| R26 | Topology-mutating actions (tie formation/dissolution) are logged as a distinct stream from content-exchange actions, since they change the graph rather than exchange information over it |

### Layer 4 — Validation & Metrics

| Rule | Requirement |
|---|---|
| R14 | A homogeneity/divergence metric (pairwise opinion distance, cluster count, dominant-share index) is defined before any full run, not after |
| R15 | Metrics are logged continuously per tick, not just at endpoint — trajectory, not just final state, is needed to diagnose *when* convergence happened |
| R16 | Maintain a null-model baseline (no persona, or non-interacting agents) to separate genuine emergent divergence from model/architecture artifacts |
| R17 | Every run is versioned against its config (architecture params + persona set + topology) so any observed convergence/divergence is traceable to the layer that caused it |
| R27 | Action-space adequacy is checked separately from homogeneity metrics — narrow action spaces can suppress observable divergence in a way the R16 null-model baseline won't catch, since the ceiling is set before the null-model comparison runs |

### Layer 5 — Interface / Query

Rules R18–R20 (listed under Layer 1 above, since they're implemented there)
conceptually belong to this layer's contract with the rest of the system.
This layer's own rule set will grow as more query types are added (forced-
choice experiments, longitudinal panels) without touching Layers 1–4.

---

## Meta-rule

Treat convergence risk — and more generally, this rule set — the way you'd
treat a cross-cutting concern in a software system, like logging or auth,
rather than a feature built once. Whenever a new component is added (a
persona field, a new interaction rule, a caching optimization), check
whether it touches shared state, sampling, or topology. If yes, it belongs
on this list before it ships.