"""Data generator for the Phase 1 model-capability sweep (ADR 0005).

Companion to phase1_model_sweep.ipynb: runs the two-agent DoD across a same-family
Qwen3 size ladder under both DST wordings and writes raw per-sample rows to
data/phase1_model_sweep.csv. The notebook only *analyses* that CSV; this script
*produces* it (it makes live OpenRouter calls). Records the chosen option *index*
(0=permDST, 1=permStandard, 2=keepSwitching, 3=noPref; -1=unparseable) so the
annotated and plain wordings stay comparable.

Run from the repo root:  python notebooks/experiments/run_model_sweep.py [K]
(K samples per grid cell; default 15. Grid = 3 models x 2 wordings x 3 conditions.)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from polis.agent import Agent
from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig, LLMError
from polis.memory_seeds import EVENING_SEEDS, MORNING_SEEDS, SHARED_PERSONA, build_store
from polis.questions import DST_QUESTION, DST_QUESTION_PLAIN

MODELS = ["qwen/qwen3-8b", "qwen/qwen3-14b", "qwen/qwen3-32b"]  # dense; no Qwen3-72B
WORDINGS = {"annotated": DST_QUESTION, "plain": DST_QUESTION_PLAIN}
CONDITIONS = {"evening": EVENING_SEEDS, "morning": MORNING_SEEDS, "empty": []}

OUT = Path("data/phase1_model_sweep.csv")


def main(k: int) -> None:
    embedder = EmbeddingModel()  # model-independent; build once
    clients = {m: LLMClient(LLMConfig(model=m)) for m in MODELS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODELS:
        client = clients[model]
        for wname, question in WORDINGS.items():
            for cname, seeds in CONDITIONS.items():
                counts: dict[int, int] = {}
                for _ in range(k):
                    agent = Agent(SHARED_PERSONA, client, embedder=embedder,
                                  memory=build_store(embedder, seeds))
                    try:
                        choice = agent.answer(question).choice
                        idx = question.options.index(choice)
                    except (LLMError, ValueError):
                        idx = -1
                    counts[idx] = counts.get(idx, 0) + 1
                    rows.append({"model": model, "wording": wname,
                                 "condition": cname, "choice_idx": idx})
                print(f"{model:16s} {wname:9s} {cname:8s} -> {dict(sorted(counts.items()))}",
                      flush=True)
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "wording", "condition", "choice_idx"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 15)
