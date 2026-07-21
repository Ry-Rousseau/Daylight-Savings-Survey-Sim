"""Individual-decision ablation — isolate the reasoning step from social dynamics.

Each condition polls all 100 agents ALONE (0 ticks, no contagion, non-mutating probe)
and tallies the DST stance distribution. If a dial moves the distribution off the
model's prior, that dial matters. Condition B (bigger decide model) is the direct test
of the model-capability hypothesis.

Watch:  Get-Content data\phase6_ablation.progress.log -Wait -Tail 30
"""
import time
from collections import Counter

import pandas as pd

from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import load_corpus
from polis.questions import DST_OPTIONS, DST_QUESTION
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import Population

CORPUS = "data/personas_corpus_n100.json"
PROGRESS = "data/phase6_ablation.progress.log"

# One dial changed per condition (all else fixed). Add C/D/E here later.
CONDITIONS = [
    ("A_32b_control",     LLMConfig(model="qwen/qwen3-32b", reasoning=False)),
    ("B_qwenmax_noreason", LLMConfig(model="qwen/qwen3.7-max", reasoning=False)),
    ("C_qwenmax_reason",  LLMConfig(model="qwen/qwen3.7-max", reasoning=True, structured_max_tokens=3072)),
]

open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("loading corpus + embedding seed memories once…")
embedder = EmbeddingModel()
pop = Population.from_corpus(CORPUS, client=LLMClient(LLMConfig()), embedder=embedder)
log(f"population ready: {len(pop.agents)} agents")

rows = []
for name, cfg in CONDITIONS:
    t0 = time.time()
    client = LLMClient(cfg)
    for a in pop.agents:            # swap the decide model; memory unchanged (probe is non-mutating)
        a.client = client
    done = [0]
    sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0),
                      on_progress=lambda: done.__setitem__(0, done[0] + 1))
    log(f"[{name}] {cfg.model} — polling 100 agents (0-tick, non-mutating)…")
    units = [(a.persona.id, (lambda a=a: a.answer(DST_QUESTION, remember=False))) for a in pop.agents]
    outcomes = sched.map(units)

    answers = [(aid, r) for aid, r, tm in outcomes if r is not None]
    skipped = [aid for aid, r, tm in outcomes if r is None]
    dist = Counter(r.choice for _, r in answers)
    top = dist.most_common(1)[0] if dist else ("—", 0)
    n = sum(dist.values())
    row = {
        "condition": name, "model": cfg.model, "n_answered": n, "n_skipped": len(skipped),
        "distinct_stances": len(dist), "dom_share": round(top[1] / n, 3) if n else 0.0,
        "top_stance": top[0][:34], "top_count": top[1], "secs": round(time.time() - t0),
    }
    for i, opt in enumerate(DST_OPTIONS):
        row[f"opt{i}"] = dist.get(opt, 0)
    rows.append(row)
    log(f"[{name}] done {row['secs']}s | answered {n} skipped {len(skipped)} | "
        f"distinct {len(dist)} | dom_share {row['dom_share']:.2f}")
    for opt in DST_OPTIONS:
        log(f"    {dist.get(opt,0):3d}  {opt[:58]}")

df = pd.DataFrame(rows)
df.to_csv("data/phase6_ablation.csv", index=False)
log("=" * 70)
log("ABLATION SUMMARY (individual, 0-tick):")
for _, r in df.iterrows():
    log(f"  {r['condition']:16s} {r['model']:22s} | distinct {r['distinct_stances']} | "
        f"dom_share {r['dom_share']:.2f} | [{r['opt0']},{r['opt1']},{r['opt2']},{r['opt3']}]")
log("opt order: [permanent-DST, permanent-standard, keep-switching, no-preference]")
log("wrote data/phase6_ablation.csv")
