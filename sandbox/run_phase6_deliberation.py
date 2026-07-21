"""Part B A/B: does removing per-turn vote-broadcasting reduce vote-convergence?

Same Sonnet population + graph, two discourse modes:
  - broadcast:  agents SPEAK stances each tick (the vote-contagion baseline).
  - deliberate: agents only SHARE_CONSIDERATION (reasons, no stance); the vote is
                read once, at the endpoint survey.
Compare the endpoint survey distribution. If deliberate stays split where broadcast
collapses, it's the vote-broadcasting — not interaction per se — that homogenizes.

Watch:  Get-Content data\phase6_deliberation.progress.log -Wait -Tail 40
"""
import time
from collections import Counter

import pandas as pd

from polis import metrics
from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import load_corpus
from polis.questions import DST_OPTIONS, DST_QUESTION
from polis.runlog import RunLog
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import DynamicsConfig, Population, Simulation
from polis.topology import SmallWorld

SEED, TICKS = 42, 5
MODEL = "anthropic/claude-sonnet-5"
CORPUS = "data/personas_corpus_n100.json"
PROGRESS = "data/phase6_deliberation.progress.log"


def sh(s):
    return {DST_OPTIONS[0]: "DST", DST_OPTIONS[1]: "STD", DST_OPTIONS[2]: "SWITCH", DST_OPTIONS[3]: "NOPREF"}.get(s, s[:8])


open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("loading corpus + embedder…")
embedder = EmbeddingModel()
client = LLMClient(LLMConfig(model=MODEL, reasoning=None))
persona_corpus = load_corpus(CORPUS)

rows = []
for mode in ("broadcast", "deliberate"):
    t0 = time.time()
    pop = Population.from_corpus(persona_corpus, client=client, embedder=embedder)
    done = [0]
    sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0),
                      on_progress=lambda: done.__setitem__(0, done[0] + 1))
    sim = Simulation(pop, topology=SmallWorld(k=4, p=0.1, seed=SEED),
                     dynamics=DynamicsConfig(seed=SEED, discourse_mode=mode),
                     logger=RunLog.open(f"data/phase6_deliberation_{mode}.sqlite"), scheduler=sched)
    log(f"[{mode}] running 100 agents x {TICKS} ticks…")
    run = sim.run(TICKS, on_tick=lambda t, n, e: log(f"[{mode}] tick {t+1}/{TICKS} | {n} decides | {e:.0f}s"))

    adq = metrics.action_space_adequacy(run, stances=list(DST_OPTIONS))
    log(f"[{mode}] action usage: speak {adq['n_speak']} | consider {adq['n_consider']} | "
        f"rebut {adq.get('n_rebut',0)} | abstain {adq['n_abstain']}")

    answers, skipped = pop.survey(DST_QUESTION, return_skipped=True)
    dist = Counter(sh(a["choice"]) for a in answers)
    n = sum(dist.values())
    top = dist.most_common(1)[0]
    row = {"mode": mode, "n": n, "dom_share": round(top[1] / n, 3), "top": top[0],
           "DST": dist.get("DST", 0), "STD": dist.get("STD", 0), "SWITCH": dist.get("SWITCH", 0),
           "NOPREF": dist.get("NOPREF", 0), "secs": round(time.time() - t0)}
    rows.append(row)
    log(f"[{mode}] endpoint survey ({n}/100): DST {row['DST']} | STD {row['STD']} | "
        f"SWITCH {row['SWITCH']} | NOPREF {row['NOPREF']}  -> dom_share {row['dom_share']:.2f}")

pd.DataFrame(rows).to_csv("data/phase6_deliberation.csv", index=False)
log("=" * 70)
log("A/B RESULT (endpoint survey; lower dom_share = less converged):")
for r in rows:
    log(f"  {r['mode']:11s} dom_share {r['dom_share']:.2f} | DST {r['DST']:3d} STD {r['STD']:3d} "
        f"SWITCH {r['SWITCH']:3d} NOPREF {r['NOPREF']:3d}")
log("wrote data/phase6_deliberation.csv")
