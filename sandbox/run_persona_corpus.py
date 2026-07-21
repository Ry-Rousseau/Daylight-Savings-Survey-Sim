"""Full persona-corpus generation (P6a) — the one-shot, cached build.

Mirrors notebooks/engineering/persona_pipeline.qmd but as a plain runner. Per-persona
retry makes the ~200-call run robust to a transient endpoint error mid-way.
backstory: qwen/qwen3-235b-a22b (0.9) · reflection: qwen/qwen3.7-max (0.3).
"""
import time

import pandas as pd

from polis.llm import LLMClient, LLMConfig
from polis.persona_pipeline import build_persona, save_corpus

DATASET = "data/persona_seed_dataset_n100.csv"
OUTPUT = "data/personas_corpus_n100.json"
SEED = 42

rows = pd.read_csv(DATASET).to_dict("records")
bs = LLMClient(LLMConfig(model="qwen/qwen3-235b-a22b", reasoning=False, max_tokens=768))
rf = LLMClient(LLMConfig(model="qwen/qwen3.7-max", reasoning=False, max_tokens=1024))


def build_with_retry(row, attempts=3):
    last = None
    for k in range(attempts):
        try:
            return build_persona(row, backstory_client=bs, reflection_client=rf, seed=SEED), None
        except Exception as e:  # noqa: BLE001 - one-shot run, retry broadly then skip
            last = repr(e)
            time.sleep(2 * (k + 1))
    return None, last


t0 = time.perf_counter()
corpus, failures = [], []
for i, row in enumerate(rows):
    sp, err = build_with_retry(row)
    if sp is None:
        failures.append((row.get("person_id"), err))
        print(f"[{i+1}/100] FAILED {row.get('person_id')}: {err}", flush=True)
        continue
    corpus.append(sp)
    if (i + 1) % 10 == 0:
        print(f"[{i+1}/100] ok ({time.perf_counter()-t0:.0f}s elapsed)", flush=True)

artifact = save_corpus(
    corpus, OUTPUT,
    meta={"seed": SEED, "dataset": DATASET,
          "backstory_model": "qwen/qwen3-235b-a22b", "reflection_model": "qwen/qwen3.7-max"},
)

empty = [sp.persona.id for sp in corpus if not sp.persona.values and not sp.persona.dispositions]
resolved = sum(1 for sp in corpus if sp.provenance["data_flags"]["resolved"])
flagged = sum(1 for sp in corpus if sp.provenance["data_flags"]["flagged"])
print("=" * 70)
print(f"DONE in {time.perf_counter()-t0:.0f}s")
print(f"wrote {OUTPUT}")
print(f"personas: {len(corpus)}/100 | failures: {len(failures)}")
print(f"content_hash: {artifact['meta']['content_hash']}")
print(f"empty-anchor: {len(empty)} {empty}")
print(f"data_flags: {resolved} with baked-out hours, {flagged} with preserved religion tension")
if failures:
    print("FAILURES:", failures)
