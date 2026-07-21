"""Label the tidy tweet corpus with topic-relevance + DST stance.

Reuses two things already built for the persona survey rather than inventing
a new taxonomy or a new structured-output path:
  - `polis.questions.DST_OPTIONS` — the same 4 stance categories the persona
    survey itself uses, so tweet labels and persona survey answers are
    directly comparable later (same categorical vocabulary).
  - `polis.llm.LLMClient.choose()` — single-select, schema-guided, retried.

One LLM call per tweet: 5-way single-select among the 4 DST_OPTIONS plus an
off-topic option. `on_topic=False` -> stance is None.

Checkpointed: results are appended to a JSONL file as they complete, and a
rerun skips ids already present in that file, so a network hiccup partway
through doesn't lose completed work.

Usage:
  Pilot (default, cheap sanity check):
    .venv/Scripts/python.exe sandbox/label_tweet_stance.py --n 15
  Full run:
    .venv/Scripts/python.exe sandbox/label_tweet_stance.py --all
"""
from __future__ import annotations

import argparse
import json
import sys
import threading

try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from polis.llm import LLMClient, LLMError  # noqa: E402
from polis.questions import DST_OPTIONS  # noqa: E402

TIDY_CSV = ROOT / "data" / "processed" / "tweets_tidy.csv"
CHECKPOINT_JSONL = ROOT / "data" / "processed" / "tweets_stance_labels.checkpoint.jsonl"
LABELED_CSV = ROOT / "data" / "processed" / "tweets_labeled.csv"

OFF_TOPIC = "Not about US daylight saving time policy (off-topic tweet)"
LABEL_OPTIONS = DST_OPTIONS + [OFF_TOPIC]

SYSTEM_PROMPT = (
    "You are labeling tweets for a research dataset on US public opinion about "
    "daylight saving time (DST) policy. Given one tweet's text, decide which "
    "position it expresses. If the tweet does not express a position on DST "
    "policy at all (e.g. unrelated news, weather reports mentioning daylight "
    "hours, other uses of the phrase), choose the off-topic option. Judge only "
    "the stance actually expressed in the text, not what a typical person might "
    "believe."
)

USER_TEMPLATE = (
    "Tweet:\n\"\"\"\n{text}\n\"\"\"\n\n"
    "Which position on US daylight saving time policy does this tweet express?"
)


def load_checkpoint() -> dict[str, dict]:
    if not CHECKPOINT_JSONL.exists():
        return {}
    done = {}
    with open(CHECKPOINT_JSONL, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            done[rec["id"]] = rec
    return done


def label_one(client: LLMClient, tweet_id: str, text: str) -> dict:
    try:
        result = client.choose(
            system=SYSTEM_PROMPT,
            user=USER_TEMPLATE.format(text=text),
            options=LABEL_OPTIONS,
            temperature=0.0,
        )
        choice = result["choice"]
        on_topic = choice != OFF_TOPIC
        return {
            "id": tweet_id,
            "on_topic": on_topic,
            "stance": choice if on_topic else None,
            "stance_index": DST_OPTIONS.index(choice) if on_topic else None,
            "stance_reason": result["reason"],
            "stance_model": result["model"],
            "error": None,
        }
    except LLMError as e:
        return {
            "id": tweet_id, "on_topic": None, "stance": None, "stance_index": None,
            "stance_reason": None, "stance_model": None, "error": str(e),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="pilot sample size (ignored with --all)")
    ap.add_argument("--all", action="store_true", help="label the full corpus")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(
        TIDY_CSV,
        dtype={"id": str, "conversation_id": str, "author_id": str,
               "in_reply_to_id": str, "in_reply_to_user_id": str, "quote_id": str},
    )

    if args.all:
        target = df
    else:
        target = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    done = load_checkpoint()
    todo = target[~target["id"].isin(done.keys())]
    print(f"Target rows: {len(target)}  already checkpointed: {len(target) - len(todo)}  to label: {len(todo)}")

    client = LLMClient()
    lock = threading.Lock()
    checkpoint_fh = open(CHECKPOINT_JSONL, "a", encoding="utf-8")

    def run(row):
        return label_one(client, row["id"], row["text"])

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, row): row["id"] for _, row in todo.iterrows()}
        for fut in as_completed(futures):
            rec = fut.result()
            with lock:
                checkpoint_fh.write(json.dumps(rec) + "\n")
                checkpoint_fh.flush()
            done[rec["id"]] = rec
            completed += 1
            if completed % 25 == 0 or completed == len(todo):
                print(f"  labeled {completed}/{len(todo)}")

    checkpoint_fh.close()

    # assemble output for the target slice
    labels_df = pd.DataFrame([done[tid] for tid in target["id"] if tid in done])
    out = target.merge(labels_df, on="id", how="left")

    if args.all:
        out.to_csv(LABELED_CSV, index=False, encoding="utf-8")
        print(f"\nWrote {LABELED_CSV} ({len(out)} rows)")

    n_errors = out["error"].notna().sum()
    n_on_topic = out["on_topic"].sum()
    print(f"\non_topic: {n_on_topic}/{len(out)}   errors: {n_errors}")
    if n_on_topic:
        print("stance distribution (on-topic only):")
        print(out.loc[out["on_topic"] == True, "stance"].value_counts().to_string())

    print("\n--- sample rows ---")
    cols = ["id", "text", "on_topic", "stance", "stance_reason"]
    with pd.option_context("display.max_colwidth", 90, "display.width", 160):
        print(out[cols].to_string(index=False))


if __name__ == "__main__":
    main()
