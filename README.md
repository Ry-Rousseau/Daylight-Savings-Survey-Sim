# Persona simulator + generator pipeline - seeding from real data

A surveyable **silicon sample of United States** — ~100 census-seeded LLM personas, polled with single-select survey questions on **daylight saving time**, to estimate what the real NYC population thinks. The engineering focus is treating opinion **convergence** (loss of realistic diversity) as a layered, cross-cutting risk — decided at each layer boundary, not patched at the end.

- **The object:** `src/polis/` — a source-first opinion-simulation engine. See `docs/design/polis-object.md`.
- **How we work:** `docs/conventions.md` (house style) · `docs/ARCHITECTURE.md` (the R-number architecture rules) · `CLAUDE.md` (routing).
- **Where we are:** `docs/status.md`.

## Phase map   

Built as a **walking skeleton, thickened one axis of complexity at a time** — every phase boundary is a runnable checkpoint isolating exactly one new unknown. Full source of truth (spike questions, layers touched, rules activated): `docs/PHASE_PLAN.md`.

| Phase | Deliverable | Definition of done |
|---|---|---|
| **0 — Walking skeleton** | 3 hardcoded agents, one survey question, structured output only | 3 agents answer 1 question; 3 parseable responses returned |
| **1 — Memory** | Vector store + recency/importance/relevance scoring; seeded memories | 2 agents with different memories give measurably different answers |
| **2 — Game Master / interaction** | Symbolic action-resolution layer; world-state store separate from memory | 2 agents complete one resolved interaction; memory + world state update consistently |
| **3 — Scheduling & throughput** | Concurrent per-agent-per-tick scheduler; latency/token/cost logged (managed backend — local GPU batching deferred to P5, ADR 0009) | 20–30 agents run multiple ticks with logged latency/cost |
| **4 — Topology** | Pluggable/swappable interaction graph | Same personas, different topology → measurably different homogeneity metric |
| **5 — Persona depth + validation** | Value-anchored personas, drift probes, R14–R17 metrics per tick | Validation dashboard running against a null-model baseline |
| **6 — Scale to full population** | Push N to 100; re-run infra benchmarks | Full 100-agent run completes with clean logs |
| **7 — Survey subsystem maturity** | Multi-timepoint surveys; response→memory writeback; calibration check | Same survey at two timepoints; differences attributable to logged events |
| **8 — Interactive GUI** *(optional/stretch)* | Thin sliders-and-run front-end (Streamlit/Gradio) over the existing config fields + live divergence plot; UI calls `src/polis`, adds no engine logic | Non-technical user runs a sim from sliders; every knob maps to a logged run-config field |

Phases are re-planned deliberately via ADRs in `docs/adr/`, never by drift.

Simulations were still in dev, whereas the persona pipeline was finished on project end.

## Quickstart (dev)

Python **3.11** via the project virtualenv (`.venv`, pinned in `.python-version`) — run with `.venv/Scripts/python.exe` or activate it.

**GPU embeddings (Phase 1+):** the memory layer embeds locally on the GPU (`sentence-transformers`, `BAAI/bge-small-en-v1.5`). `torch` must come from the CUDA wheel index — a bare install yields the CPU build:

```
uv pip install --python .venv/Scripts/python.exe torch --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev,notebooks]"
```

Confirm with `.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available())"` → `True`.

LLM backend is a **remote** vLLM server (`Qwen/Qwen3-8B-AWQ`) — see `CLAUDE.md` and `docs/query_handbook.md`. For notebooks, register the venv as a Jupyter kernel once (`.venv/Scripts/python.exe -m ipykernel install --user --name polis --display-name "Python 3.11 (polis .venv)"`) and select it in the kernel picker.
