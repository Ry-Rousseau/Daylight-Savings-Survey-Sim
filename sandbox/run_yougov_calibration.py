"""YouGov calibration — ask the silicon sample the REAL survey questions (Q2, Q4) and
score each model against the published toplines. The strongest external validity test:
do the agents reason toward what real Americans actually answered?

Watch:  Get-Content data\phase6_calibration.progress.log -Wait -Tail 40
"""
import time
from collections import Counter

import pandas as pd

from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import load_corpus
from polis.questions import (
    YOUGOV_Q2_ELIMINATE, YOUGOV_Q2_TOPLINE,
    YOUGOV_Q4_PERMANENT, YOUGOV_Q4_TOPLINE,
)
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import Population

PROGRESS = "data/phase6_calibration.progress.log"
OUT_CSV = "data/phase6_calibration_sonnet.csv"
CONDITIONS = [
    ("claude_sonnet5", LLMConfig(model="anthropic/claude-sonnet-5", reasoning=None)),
]
QUESTIONS = [("Q2_eliminate", YOUGOV_Q2_ELIMINATE, YOUGOV_Q2_TOPLINE),
             ("Q4_permanent", YOUGOV_Q4_PERMANENT, YOUGOV_Q4_TOPLINE)]

open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def tvd(dist_frac, topline):
    """Total-variation distance (0 = perfect match, 1 = disjoint) vs the topline."""
    keys = set(dist_frac) | set(topline)
    return round(0.5 * sum(abs(dist_frac.get(k, 0.0) - topline.get(k, 0.0)) for k in keys), 3)


log("loading corpus + embedder…")
embedder = EmbeddingModel()
pop = Population.from_corpus("data/personas_corpus_n100.json", client=LLMClient(LLMConfig()), embedder=embedder)

rows = []
for cname, cfg in CONDITIONS:
    client = LLMClient(cfg)
    for a in pop.agents:
        a.client = client
    for qname, q, topline in QUESTIONS:
        t0 = time.time()
        sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0))
        units = [(a.persona.id, (lambda a=a, q=q: a.answer(q, remember=False))) for a in pop.agents]
        out = sched.map(units)
        answers = [r.choice for _, r, t in out if r is not None]
        n = len(answers)
        dist = Counter(answers)
        frac = {k: round(v / n, 3) for k, v in dist.items()} if n else {}
        gap = tvd(frac, topline)
        row = {"model": cname, "question": qname, "n": n, "tvd_vs_yougov": gap, "secs": round(time.time() - t0)}
        rows.append(row)
        log(f"[{cname} · {qname}] n={n} | TVD vs YouGov = {gap} | {row['secs']}s")
        for opt in q.options:
            log(f"    model {frac.get(opt,0.0)*100:4.0f}%  vs  YouGov {topline.get(opt,0)*100:4.0f}%   {opt[:52]}")

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
log("=" * 72)
log("CALIBRATION SUMMARY (total-variation distance from YouGov; lower = better):")
piv = df.pivot(index="model", columns="question", values="tvd_vs_yougov")
for m in piv.index:
    log(f"  {m:16s}  Q2 TVD {piv.loc[m,'Q2_eliminate']:.3f}  |  Q4 TVD {piv.loc[m,'Q4_permanent']:.3f}")
log("wrote data/phase6_calibration.csv")
