"""Attractor-flip test: seed a DST plurality, amplify it on a dense (~50%) random graph.

Hypothesis: the cascade amplifies whatever leads at t=0. Standard won before only because
it started ahead. Conviction-seed DST into the lead, run a dense graph 3 ticks, and the
same rich-get-richer dynamic should snowball DST instead.

Watch:  Get-Content data\phase6_dst_flip.progress.log -Wait -Tail 40
"""
import time
from collections import Counter

import pandas as pd

from polis import metrics
from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.opinion_seeds import OpinionPlan, apply_opinion_plan, load_opinion_corpus
from polis.persona_pipeline import load_corpus
from polis.questions import DST_OPTIONS, DST_QUESTION
from polis.runlog import RunLog
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import DynamicsConfig, Population, Simulation
from polis.topology import StochasticBlock

SEED, TICKS = 42, 3
MODEL = "anthropic/claude-sonnet-5"
PROGRESS = "data/phase6_dst_flip.progress.log"
DST, STD = DST_OPTIONS[0], DST_OPTIONS[1]


def sh(s):  # short stance label
    return {DST: "DST", STD: "STD", DST_OPTIONS[2]: "SWITCH", DST_OPTIONS[3]: "NOPREF"}.get(s, s[:8])


open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # Windows cp1252 console can't encode some unicode
        print(line.encode("ascii", "replace").decode("ascii"), flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("loading corpus + opinion corpus + embedder…")
embedder = EmbeddingModel()
client = LLMClient(LLMConfig(model=MODEL, reasoning=None))
pop = Population.from_corpus(load_corpus("data/personas_corpus_n100.json"), client=client, embedder=embedder)
opinion_corpus = load_opinion_corpus()

# Conviction-seed 40% of agents as strong DST advocates -> tip the tick-0 plurality to DST.
plan = OpinionPlan(scheme="two_camp", camps=((DST, 0.40),), phrasing="conviction",
                   n_opinions=2, importance=8.0, seed=SEED)
prov = apply_opinion_plan(pop, opinion_corpus, plan)
log(f"seeded {prov['n_seeded']}/100 agents with DST conviction (phrasing=conviction)")

done = [0]
sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0),
                  on_progress=lambda: done.__setitem__(0, done[0] + 1))
sim = Simulation(
    pop, topology=StochasticBlock(n_blocks=1, p_in=0.5, seed=SEED),  # ~50% random connectivity
    dynamics=DynamicsConfig(seed=SEED),
    logger=RunLog.open("data/phase6_dst_flip.sqlite"),
    scheduler=sched,
)
log(f"START: 100 agents x {TICKS} ticks, stochastic_block(1, p_in=0.5 ~= 50% connected), {MODEL}")
run = sim.run(TICKS, on_tick=lambda t, n, e: log(f"tick {t+1}/{TICKS} complete | {n} decides | {e:.0f}s"))
tp = run.throughput
log(f"RUN DONE {tp['wall_s']:.0f}s | {tp['n_calls']} decides | {tp['failures']} failures")

log("computing metrics…")
adq = metrics.action_space_adequacy(run, stances=list(DST_OPTIONS))
log(f"R27: abstain {adq['abstain_rate']:.2f} | stances_used {adq['distinct_stances_used']}/4 | flags {adq['flags'] or 'none'}")

log("--- per-tick stance breakdown (watch DST snowball) ---")
for t in range(TICKS):
    d = Counter({sh(k): v for k, v in metrics.stance_distribution(run, tick=t).items()})
    tot = sum(d.values())
    top = d.most_common(1)[0]
    log(f"  tick {t}: DST {d.get('DST',0):3d} | STD {d.get('STD',0):3d} | SWITCH {d.get('SWITCH',0):3d} | "
        f"NOPREF {d.get('NOPREF',0):3d}   -> winner {top[0]} ({top[1]}/{tot})")

pd.DataFrame(metrics.divergence_trajectory(run, embedder, support=len(DST_OPTIONS))).to_csv(
    "data/phase6_dst_flip_trajectory.csv", index=False)

log("endpoint survey…")
answers, skipped = pop.survey(DST_QUESTION, return_skipped=True)
dist = Counter(sh(a["choice"]) for a in answers)
log(f"survey {len(answers)}/100 (skipped {skipped or 'none'}): "
    f"DST {dist.get('DST',0)} | STD {dist.get('STD',0)} | SWITCH {dist.get('SWITCH',0)} | NOPREF {dist.get('NOPREF',0)}")
log("DONE")
