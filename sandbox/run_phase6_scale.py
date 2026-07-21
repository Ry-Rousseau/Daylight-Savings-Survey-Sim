"""Phase 6b live run (1-arm, full corpus, 5 ticks) with streamed progress.

Watch it live:  Get-Content data\phase6_scale.progress.log -Wait -Tail 30
Mirrors notebooks/experiments/phase6_scale.qmd; writes the durable log + metric CSVs.
"""
import time

import pandas as pd

from polis import metrics
from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.questions import DST_OPTIONS, DST_QUESTION
from polis.runlog import RunLog
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import DynamicsConfig, Population, Simulation
from polis.topology import RingLattice, SmallWorld

SEED, TICKS = 42, 5
CORPUS = "data/personas_corpus_n100.json"
TAG = "sonnet_ring"  # output suffix so this run doesn't clobber the qwen baseline
MODEL = "anthropic/claude-sonnet-5"
PROGRESS = f"data/phase6_scale_{TAG}.progress.log"
SUPPORT = len(DST_OPTIONS)

open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("loading corpus + embedding seed memories (100 agents)…")
embedder = EmbeddingModel()
client = LLMClient(LLMConfig(model=MODEL, reasoning=None))
pop = Population.from_corpus(CORPUS, client=client, embedder=embedder)
log(f"population ready: {len(pop.agents)} agents | corpus {pop.corpus_meta.get('content_hash','')[:12]}")

done = [0]


def on_progress():
    done[0] += 1
    if done[0] % 25 == 0:
        log(f"  …{done[0]} decides completed")


def on_tick(t, ncalls, elapsed):
    log(f"tick {t+1}/{TICKS} complete | {ncalls} decides total | {elapsed:.0f}s elapsed")


sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0),
                  on_progress=on_progress)
sim = Simulation(
    pop, topology=RingLattice(k=2, seed=SEED),
    dynamics=DynamicsConfig(seed=SEED),
    logger=RunLog.open(f"data/phase6_scale_{TAG}.sqlite"),
    scheduler=sched,
)

log(f"START run: {len(pop.agents)} agents × {TICKS} ticks, ring_lattice(k=2), {MODEL}")
run = sim.run(TICKS, on_tick=on_tick)
tp = run.throughput
log(f"RUN DONE in {tp['wall_s']:.0f}s | {tp['n_calls']} decides | {tp['decides_per_s']:.2f} decides/s | "
    f"p95 {tp['latency_p95_s']:.1f}s | retries {tp['retries']} | failures {tp['failures']}")

# --- metrics (offline, no LLM) ---
log("computing metrics…")
adq = metrics.action_space_adequacy(run, stances=list(DST_OPTIONS), embedder=embedder)
log(f"R27 gate: abstain_rate {adq['abstain_rate']:.2f} | stances_used {adq['distinct_stances_used']}/4 | "
    f"utterance_uniqueness {adq['utterance_uniqueness']:.2f} | flags {adq['flags'] or 'none'}")

traj = pd.DataFrame(metrics.divergence_trajectory(run, embedder, support=SUPPORT))
traj.to_csv(f"data/phase6_scale_{TAG}_trajectory.csv", index=False)
for _, r in traj.iterrows():
    log(f"  tick {int(r['tick'])}: dom_share {r['dominant_share']:.2f} | entropy {r['entropy']:.2f} | "
        f"dispersion {r['pairwise_dispersion']:.3f} | clusters {int(r['cluster_count'])} (n={int(r['n_utterances'])})")

summary = metrics.divergence_summary(run, embedder, support=SUPPORT)
log(f"ENDPOINT: dom_share {summary['dominant_share']:.2f} | dispersion {summary['pairwise_dispersion']:.3f} | "
    f"clusters {summary['cluster_count']}/{summary['n_utterances']}")

log("running endpoint survey (100 agents)…")
answers, skipped = pop.survey(DST_QUESTION, return_skipped=True)
dist = pd.Series([a["choice"] for a in answers]).value_counts()
dist.to_csv(f"data/phase6_scale_{TAG}_survey.csv")
log(f"survey: {len(answers)}/{len(pop.agents)} answered (skipped {skipped or 'none'})")
for opt, ct in dist.items():
    log(f"  {ct:3d}  {opt[:60]}")
log(f"DONE — wrote phase6_scale_{TAG}_trajectory.csv, phase6_scale_{TAG}_survey.csv, phase6_scale_{TAG}.sqlite")
