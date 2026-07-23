"""Per-persona stance survey: query each of the 100 Sonnet personas individually on the
YouGov Q4 (permanent DST vs standard), save the answer joined to its demographics.
Feeds the 'what predicts stance?' analysis. ~100 calls."""
import time

import pandas as pd

from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import load_corpus
from polis.questions import YOUGOV_Q4_PERMANENT
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import Population

MODEL = "anthropic/claude-sonnet-5"
Q = YOUGOV_Q4_PERMANENT


def sh(s):
    return {Q.options[0]: "DST", Q.options[1]: "Standard", Q.options[2]: "No preference",
            Q.options[3]: "Not sure"}.get(s, s[:12])


t0 = time.time()
emb = EmbeddingModel()
pop = Population.from_corpus(load_corpus("data/personas_corpus_n100.json"),
                            client=LLMClient(LLMConfig(model=MODEL, reasoning=None)), embedder=emb)
units = [(a.persona.id, (lambda a=a: a.answer(Q, remember=False))) for a in pop.agents]
out = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4)).map(units)
ans = {aid: sh(r.choice) for aid, r, tm in out if r is not None}
# Keep the free-text rationale too — it is the "why" behind each choice and the only
# place the individual survey records voice; dropping it (as the first version did)
# leaves nothing to inspect for Layer 1. `stance` stays the short code for existing analysis.
reasons = {aid: r.reason for aid, r, tm in out if r is not None}
print(f"answered {len(ans)}/100 in {time.time()-t0:.0f}s")

demo = pd.read_csv("data/persona_seed_dataset_n100.csv")
demo["stance"] = demo["person_id"].map(ans)
demo["stance_reason"] = demo["person_id"].map(reasons)
demo = demo.dropna(subset=["stance"])
demo.to_csv("data/phase6_persona_stances.csv", index=False)
print("wrote data/phase6_persona_stances.csv")
print(demo["stance"].value_counts().to_string())
