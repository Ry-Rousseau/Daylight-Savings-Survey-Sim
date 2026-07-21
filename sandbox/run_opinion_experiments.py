"""Opinion-seeding experiment matrix — the conviction-slider sweep.

Each run: fresh 100-persona population from the corpus, apply an OpinionPlan, run the sim,
read the endpoint divergence. Edit EXPERIMENTS and re-run. ~2 min each (survey skipped —
the log endpoint dom_share IS the vote proxy). Compare in data/phase6_opinion_experiments.csv.

Watch:  Get-Content data\phase6_opinion.progress.log -Wait -Tail 30
"""
import time

import pandas as pd

from polis import metrics
from polis.embeddings import EmbeddingModel
from polis.llm import LLMClient, LLMConfig
from polis.opinion_seeds import OpinionPlan, apply_opinion_plan, load_opinion_corpus
from polis.persona_pipeline import load_corpus
from polis.questions import DST_OPTIONS
from polis.runlog import RunLog
from polis.scheduler import Scheduler, SchedulerConfig
from polis.simulation import DynamicsConfig, Population, Simulation
from polis.topology import SmallWorld

SEED, TICKS, SUPPORT = 42, 5, len(DST_OPTIONS)
CORPUS = "data/personas_corpus_n100.json"
PROGRESS = "data/phase6_opinion.progress.log"
DST, STD = DST_OPTIONS[0], DST_OPTIONS[1]

# ---- the matrix — edit freely -------------------------------------------------
EXPERIMENTS = [
    ("baseline",            OpinionPlan(scheme="none")),
    ("polar25_seen",        OpinionPlan(scheme="two_camp", camps=((DST, 0.25), (STD, 0.25)), phrasing="seen")),
    ("polar50_seen",        OpinionPlan(scheme="two_camp", camps=((DST, 0.50), (STD, 0.50)), phrasing="seen")),
    ("polar25_conviction",  OpinionPlan(scheme="two_camp", camps=((DST, 0.25), (STD, 0.25)), phrasing="conviction")),
    ("polar25_committed",   OpinionPlan(scheme="two_camp", camps=((DST, 0.25), (STD, 0.25)), phrasing="seen", committed=(DST, STD))),
    ("random50_seen",       OpinionPlan(scheme="random", fraction=0.50, stances=(DST, STD), phrasing="seen")),
]

open(PROGRESS, "w").close()


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(PROGRESS, "a", encoding="utf-8") as f:
        f.write(line + "\n")


log("loading persona corpus + opinion corpus + embedder…")
persona_corpus = load_corpus(CORPUS)
opinion_corpus = load_opinion_corpus()
embedder = EmbeddingModel()
client = LLMClient(LLMConfig(model="qwen/qwen3-32b", reasoning=False))
log(f"opinion poles available: DST={len(opinion_corpus.get(DST, []))} STD={len(opinion_corpus.get(STD, []))}")

rows = []
for name, plan in EXPERIMENTS:
    t0 = time.time()
    pop = Population.from_corpus(persona_corpus, client=client, embedder=embedder)
    prov = apply_opinion_plan(pop, opinion_corpus, plan)
    done = [0]
    sched = Scheduler(SchedulerConfig(max_concurrency=16, max_retries=4, backoff_base=1.0, backoff_max=15.0),
                      on_progress=lambda: (done.__setitem__(0, done[0] + 1)))
    sim = Simulation(pop, topology=SmallWorld(k=4, p=0.1, seed=SEED),
                     dynamics=DynamicsConfig(seed=SEED), logger=RunLog.open(":memory:"), scheduler=sched)
    log(f"[{name}] seeded {prov['n_seeded']}/100 (committed {prov.get('n_committed',0)}) — running…")
    run = sim.run(TICKS)

    traj = metrics.divergence_trajectory(run, embedder, support=SUPPORT)
    pd.DataFrame(traj).to_csv(f"data/phase6_opinion_{name}_trajectory.csv", index=False)
    end = traj[-1]
    dist = metrics.stance_distribution(run)  # endpoint stance tally
    top = dist.most_common(1)[0] if dist else ("—", 0)
    adq = metrics.action_space_adequacy(run, stances=list(DST_OPTIONS))
    rows.append({
        "experiment": name, "scheme": plan.scheme, "phrasing": plan.phrasing,
        "n_seeded": prov["n_seeded"], "n_committed": prov.get("n_committed", 0),
        "end_dom_share": round(end["dominant_share"], 3), "end_entropy": round(end["entropy"], 3),
        "end_dispersion": round(end["pairwise_dispersion"], 3), "end_clusters": end["cluster_count"],
        "distinct_stances_end": end["distinct"], "abstain_rate": round(adq["abstain_rate"], 3),
        "top_stance": top[0][:34], "top_count": top[1], "secs": round(time.time() - t0),
    })
    log(f"[{name}] done in {rows[-1]['secs']}s | dom_share {end['dominant_share']:.2f} | "
        f"clusters {end['cluster_count']} | distinct_stances {end['distinct']} | top '{top[0][:30]}' x{top[1]}")

df = pd.DataFrame(rows)
df.to_csv("data/phase6_opinion_experiments.csv", index=False)
log("=" * 70)
log("COMPARISON (endpoint):")
for _, r in df.iterrows():
    log(f"  {r['experiment']:20s} seeded {r['n_seeded']:3d} | dom_share {r['end_dom_share']:.2f} | "
        f"clusters {r['end_clusters']:2d} | stances {r['distinct_stances_end']} | top×{r['top_count']}")
log("wrote data/phase6_opinion_experiments.csv")
