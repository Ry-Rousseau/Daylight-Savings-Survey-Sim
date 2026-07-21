"""Clean + tidy the raw X (Twitter) API dumps in data/x_api_data/.

Source data: Apify exports from two scrapers ("tweet-scraper" = search /
top-level posts, "twitter-scraper-lite" = conversation replies). Despite the
different scraper names, every record shares one tweet schema, so both are
parsed identically; the scraper name is kept only as provenance.

What this does:
  - drops the one placeholder file that has no real results ({"noResults": true})
  - recovers quote-tweets' nested `quote` object as first-class tweets
  - deduplicates by tweet id across all files/timestamps (89 ids were scraped
    more than once as engagement counters climbed); keeps the most recently
    scraped version of each and records how many times / which files it was seen in
  - flattens author + entities + media + card metadata into columns
  - writes a tidy table (CSV) for querying, plus a lossless deduped JSONL backup
    of the original nested records for anything the tidy table doesn't surface

Run: .venv/Scripts/python.exe sandbox/clean_x_data.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "x_api_data"
OUT_DIR = ROOT / "data" / "processed"

FILENAME_RE = re.compile(
    r"^dataset_(?P<scraper>.+?)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}-\d{3})\.json$"
)


def parse_filename(name: str):
    m = FILENAME_RE.match(name)
    if not m:
        raise ValueError(f"unrecognized filename pattern: {name}")
    dt = pd.to_datetime(m.group("ts"), format="%Y-%m-%d_%H-%M-%S-%f", utc=True)
    return m.group("scraper"), dt


def parse_twitter_dt(s):
    if not s:
        return pd.NaT
    try:
        return pd.to_datetime(s, format="%a %b %d %H:%M:%S %z %Y", utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def clean_text(s):
    if s is None:
        return None
    s = s.strip()
    return s or None


def join(values, sep=","):
    values = [v for v in values if v]
    return sep.join(values) if values else None


def extract_entities(entities):
    entities = entities or {}
    hashtags = [h.get("text") for h in entities.get("hashtags", []) or []]
    mentions = [m.get("screen_name") for m in entities.get("user_mentions", []) or []]
    urls = [u.get("expanded_url") or u.get("url") for u in entities.get("urls", []) or []]
    return {
        "hashtags": join(hashtags),
        "mentioned_usernames": join(mentions),
        "urls_expanded": join(urls, sep="|"),
    }


def extract_media(record):
    ext_media = ((record.get("extendedEntities") or {}).get("media")) or []
    if ext_media:
        types = sorted({m.get("type") for m in ext_media if m.get("type")})
        urls = [m.get("media_url_https") for m in ext_media if m.get("media_url_https")]
        return {"media_count": len(ext_media), "media_types": join(types), "media_urls": join(urls, sep="|")}
    flat_media = record.get("media") or []
    if flat_media:
        return {"media_count": len(flat_media), "media_types": None, "media_urls": join(flat_media, sep="|")}
    return {"media_count": 0, "media_types": None, "media_urls": None}


def extract_card(card):
    if not card:
        return {"card_title": None, "card_description": None, "card_domain": None, "card_url": None}
    legacy = card.get("legacy") or {}
    bindings = {b.get("key"): (b.get("value") or {}) for b in legacy.get("binding_values", []) or []}

    def sval(key):
        return bindings.get(key, {}).get("string_value")

    return {
        "card_title": sval("title"),
        "card_description": sval("description"),
        "card_domain": sval("domain"),
        "card_url": sval("card_url") or sval("vanity_url") or legacy.get("url"),
    }


def extract_place(place):
    if not place:
        return {"place_name": None, "place_country": None}
    return {
        "place_name": place.get("name") or place.get("full_name"),
        "place_country": place.get("country"),
    }


def flatten_author(author):
    author = author or {}
    prof = author.get("professional") or {}
    categories = prof.get("category") or []
    return {
        "author_id": author.get("id"),
        "author_username": author.get("userName"),
        "author_name": author.get("name"),
        "author_description": clean_text(author.get("description")),
        "author_location": author.get("location") or None,
        "author_created_at": parse_twitter_dt(author.get("createdAt")),
        "author_followers": author.get("followers"),
        "author_following": author.get("following"),
        "author_statuses_count": author.get("statusesCount"),
        "author_favourites_count": author.get("favouritesCount"),
        "author_media_count": author.get("mediaCount"),
        "author_is_verified": bool(author.get("isVerified", False)),
        "author_is_blue_verified": bool(author.get("isBlueVerified", False)),
        "author_protected": bool(author.get("protected", False)),
        "author_professional_type": prof.get("professional_type"),
        "author_category": join([c.get("name") for c in categories]),
    }


def flatten_tweet(record: dict) -> dict:
    row = {
        "id": record.get("id"),
        "conversation_id": record.get("conversationId"),
        "in_reply_to_id": record.get("inReplyToId"),
        "in_reply_to_user_id": record.get("inReplyToUserId"),
        "in_reply_to_username": record.get("inReplyToUsername"),
        "quote_id": record.get("quoteId"),
        "is_reply": bool(record.get("isReply", False)),
        "is_quote": bool(record.get("isQuote", False)),
        "is_retweet": bool(record.get("isRetweet", False)),
        "is_pinned": bool(record.get("isPinned", False)),
        "is_conversation_controlled": bool(record.get("isConversationControlled", False)),
        "possibly_sensitive": bool(record.get("possiblySensitive", False)),
        "created_at": parse_twitter_dt(record.get("createdAt")),
        "text": clean_text(record.get("fullText") or record.get("text") or ""),
        "lang": record.get("lang") or None,
        "source_client": record.get("source") or None,
        "like_count": record.get("likeCount"),
        "retweet_count": record.get("retweetCount"),
        "reply_count": record.get("replyCount"),
        "quote_count": record.get("quoteCount"),
        "view_count": record.get("viewCount"),
        "bookmark_count": record.get("bookmarkCount"),
        "url": record.get("url"),
        "twitter_url": record.get("twitterUrl"),
    }
    row.update(extract_entities(record.get("entities")))
    row.update(extract_media(record))
    row.update(extract_card(record.get("card")))
    row.update(extract_place(record.get("place")))
    row.update(flatten_author(record.get("author")))
    return row


def load_all_occurrences():
    """One entry per (file, record) occurrence, including quote-tweets recovered
    from their parent's nested `quote` object."""
    occurrences = []
    files = sorted(DATA_DIR.glob("*.json"))
    n_placeholder_files = 0
    n_placeholder_records = 0

    for f in files:
        scraper_type, file_dt = parse_filename(f.name)
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)

        real_in_file = 0
        for r in data:
            if not isinstance(r, dict) or set(r.keys()) == {"noResults"}:
                n_placeholder_records += 1
                continue
            real_in_file += 1
            occurrences.append(
                {"file_dt": file_dt, "file_name": f.name, "scraper_type": scraper_type,
                 "record": r, "nested_via": None}
            )
            quote = r.get("quote")
            if isinstance(quote, dict) and quote.get("id"):
                occurrences.append(
                    {"file_dt": file_dt, "file_name": f.name, "scraper_type": scraper_type,
                     "record": quote, "nested_via": r.get("id")}
                )
        if real_in_file == 0:
            n_placeholder_files += 1

    return occurrences, n_placeholder_files, n_placeholder_records


def dedupe(occurrences):
    groups = defaultdict(list)
    for occ in occurrences:
        tid = occ["record"].get("id")
        if tid:
            groups[tid].append(occ)

    rows = []
    raw_backup = []
    for tid, occs in groups.items():
        occs_sorted = sorted(occs, key=lambda o: o["file_dt"])
        best = occs_sorted[-1]["record"]  # most recently scraped -> freshest engagement counts
        files_seen = sorted({o["file_name"] for o in occs})
        scrapers_seen = sorted({o["scraper_type"] for o in occs})
        nested_parents = sorted({o["nested_via"] for o in occs if o["nested_via"]})

        row = flatten_tweet(best)
        row["n_times_scraped"] = len(occs)
        row["source_files"] = "|".join(files_seen)
        row["scraper_types"] = "|".join(scrapers_seen)
        row["seen_via_quote_of"] = join(nested_parents, sep="|")
        row["first_scraped_at"] = occs_sorted[0]["file_dt"]
        row["last_scraped_at"] = occs_sorted[-1]["file_dt"]
        rows.append(row)

        backup = dict(best)
        backup["_source_files"] = files_seen
        backup["_scraper_types"] = scrapers_seen
        backup["_n_times_scraped"] = len(occs)
        raw_backup.append(backup)

    return rows, raw_backup


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    occurrences, n_placeholder_files, n_placeholder_records = load_all_occurrences()
    n_direct = sum(1 for o in occurrences if o["nested_via"] is None)
    n_nested = sum(1 for o in occurrences if o["nested_via"] is not None)

    rows, raw_backup = dedupe(occurrences)
    df = pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)

    # tweets that only exist because they were quoted (never independently scraped)
    ids_direct = {o["record"]["id"] for o in occurrences if o["nested_via"] is None}
    only_via_quote = sum(1 for r in rows if r["id"] not in ids_direct)

    csv_path = OUT_DIR / "tweets_tidy.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    raw_backup_sorted = sorted(raw_backup, key=lambda r: r.get("createdAt") or "")
    jsonl_path = OUT_DIR / "tweets_raw_deduped.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for rec in raw_backup_sorted:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- summary report ----
    print("=" * 70)
    print("X / Twitter DST dataset — cleaning summary")
    print("=" * 70)
    print(f"Files read:                 {len(sorted(DATA_DIR.glob('*.json')))}")
    print(f"  placeholder files (empty): {n_placeholder_files}")
    print(f"  placeholder records dropped: {n_placeholder_records}")
    print(f"Raw record occurrences:     {n_direct + n_nested}")
    print(f"  direct (top-level in a file): {n_direct}")
    print(f"  recovered from quote-tweets:  {n_nested}")
    print(f"Unique tweets after dedup:  {len(df)}")
    print(f"  duplicate occurrences merged: {(n_direct + n_nested) - len(df)}")
    print(f"  tweets only ever seen as a quote (not independently scraped): {only_via_quote}")
    print()
    print(f"Date range: {df['created_at'].min()} -> {df['created_at'].max()}")
    print(f"is_reply:  True={df['is_reply'].sum()}  False={(~df['is_reply']).sum()}")
    print(f"is_quote:  True={df['is_quote'].sum()}")
    print(f"lang top 5: {Counter(df['lang'].dropna()).most_common(5)}")
    print(f"unique authors: {df['author_username'].nunique()}")
    print("top 5 authors by tweet count:")
    for name, cnt in df["author_username"].value_counts().head(5).items():
        print(f"  {name}: {cnt}")
    all_hashtags = [h for cell in df["hashtags"].dropna() for h in cell.split(",")]
    print(f"top 10 hashtags: {Counter(all_hashtags).most_common(10)}")
    print()
    print(f"Wrote tidy table -> {csv_path}  ({len(df)} rows, {len(df.columns)} cols)")
    print(f"Wrote raw backup  -> {jsonl_path}  ({len(raw_backup_sorted)} records)")
    print()
    print("Note: id / conversation_id / author_id / in_reply_to_id / quote_id are")
    print("Twitter snowflake ints too large for float64 - kept as strings in the CSV.")
    print("Reload with pandas.read_csv(..., dtype={'id': str, 'conversation_id': str,")
    print("'author_id': str, 'in_reply_to_id': str, 'in_reply_to_user_id': str, 'quote_id': str},")
    print("parse_dates=['created_at','author_created_at','first_scraped_at','last_scraped_at'])")


if __name__ == "__main__":
    main()
