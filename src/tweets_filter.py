"""One pass over the raw vendor corpus: find the climate posts and keep everything
the analysis and the referees need, so the 170 GB never has to be read twice.

Driven by 02_tweets_filter.ipynb. The worker lives in a module rather than in the
notebook because ProcessPoolExecutor has to be able to import it.

Per raw .json.gz, three files land in <outdir>:

  <stem>.tweets.csv   one row per climate post: the full text, the place, the author,
                      the language, the client app and the engagement counts
  <stem>.agg.csv      counts over ALL posts in the file, by country and language.
                      This is the denominator behind "posts / unique users / retweet
                      share / language mix per country"
  <stem>.hll.pkl.gz   HyperLogLog sketches, one per country, for distinct authors
                      over ALL posts. Mergeable across files, ~2% error

The filter runs on `extended_tweet.full_text` where the vendor truncated `text` at
140 characters. For truncated posts it is also run on the short `text`, which is what
the 2025 run saw, so the posts truncation cost us can be counted exactly rather than
estimated.
"""

import collections
import csv
import gzip
import json
import os
import pickle

import climate_filter

HLL_P = 12                 # 4096 registers per country, ~2% standard error
HLL_M = 1 << HLL_P

FIELDS = [
    "created_at", "tweet_id", "country", "country_code", "city", "place_type",
    "has_coords", "lang", "source",
    "text", "full_text", "was_truncated", "hit_on_text", "hashtags",
    "is_rt", "is_quote", "is_reply", "rt_count", "favorite_count", "reply_count",
    "quote_count",
    "user_id", "screen_name", "user_name", "verified", "user_description",
    "user_location", "followers_count", "friends_count", "statuses_count",
    "listed_count", "favourites_count", "user_created_at",
    "default_profile", "default_profile_image",
]

AGG_FIELDS = ["country_code", "lang", "n", "n_rt", "n_reply", "n_quote",
              "n_truncated", "n_verified", "n_climate", "n_climate_on_text"]

_MON = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def to_iso(s):
    """
    'Wed Dec 15 00:00:00 +0000 2021' -> '2021-12-15 00:00:00+00:00'.

    Hand-rolled because strptime is about twenty times slower and this runs a few
    hundred million times. The feed is always +0000. Checked against strptime on
    60,000 records with no disagreement.
    """
    return f"{s[26:30]}-{_MON[s[4:7]]:02d}-{s[8:10]} {s[11:19]}+00:00"


def strip_source(s):
    """'<a href="...">Twitter for iPhone</a>' -> 'Twitter for iPhone'."""
    if not s:
        return ""
    i, j = s.find(">"), s.rfind("</a>")
    return s[i + 1:j] if i != -1 and j != -1 else s


# ── HyperLogLog, for distinct authors per country over all posts ────────────
def _mix64(x):
    """splitmix64 finaliser: well distributed, and no hashlib call per post."""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 31)


def hll_add(reg, uid):
    try:
        h = _mix64(int(uid))
    except (TypeError, ValueError):
        return
    idx = h >> (64 - HLL_P)
    w = (h << HLL_P) & 0xFFFFFFFFFFFFFFFF
    rank = 1 if w else 65 - HLL_P
    while w and not (w & 0x8000000000000000):
        w <<= 1
        rank += 1
    if rank > reg[idx]:
        reg[idx] = rank


def hll_estimate(reg):
    import math
    V = reg.count(0)
    alpha = 0.7213 / (1 + 1.079 / HLL_M)
    E = alpha * HLL_M * HLL_M / sum(2.0 ** -r for r in reg)
    if E < 2.5 * HLL_M and V:
        return HLL_M * math.log(HLL_M / V)
    return E


def merge_hll(paths):
    """Merge per-file sketches into one register array per country."""
    merged = collections.defaultdict(lambda: bytearray(HLL_M))
    for p in paths:
        with gzip.open(p, "rb") as fh:
            for cc, reg in pickle.load(fh).items():
                t = merged[cc]
                for i, r in enumerate(reg):
                    if r > t[i]:
                        t[i] = r
    return {cc: round(hll_estimate(reg)) for cc, reg in merged.items()}


# ── which taxonomy rule fired ───────────────────────────────────────────────
# The Global Pulse taxonomy is a set of OR-ed rules. Some name climate directly;
# the rest are conjunctions that need a climate word plus a topical word, and those
# fire more readily on long text simply because there is more of it to match.
# Splitting them this way is what makes the precision check interpretable.
STRONG_RULES = ["matches_general", "matches_arctic", "matches_hashtag_current",
                "matches_hashtag_campaigns", "matches_hashtag_negotiations",
                "matches_hashtag_related"]
WEAK_RULES = ["matches_risk_disaster", "matches_politics_opinion", "matches_economy",
              "matches_energy", "matches_oceans_water", "matches_weather",
              "matches_agriculture_forestry"]
ALL_RULES = STRONG_RULES + WEAK_RULES


def fired_rules(text):
    """Which taxonomy rules a post satisfies. Expects lowercased text."""
    return [r for r in ALL_RULES if getattr(climate_filter, r)(text)]


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion; behaves at small n, unlike normal."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── the worker ──────────────────────────────────────────────────────────────
def process(path, outdir, year=None, require_place=True):
    """
    Scan one raw .json.gz and write the three output files described above.

    year          keep only posts from this calendar year (None keeps all)
    require_place keep only posts the vendor geotagged, which is what the
                  country-level analysis needs

    Returns (stem, posts_scanned, climate_posts_kept, unparsable_lines).
    """
    stem = os.path.basename(path).replace(".json.gz", "").replace(".json", "")
    opener = gzip.open if path.endswith(".gz") else open

    rows = []
    agg = collections.defaultdict(lambda: [0] * 8)   # see AGG_FIELDS[2:]
    hll = collections.defaultdict(lambda: bytearray(HLL_M))
    total = bad = 0

    with opener(path, "rb") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except Exception:
                bad += 1
                continue
            total += 1

            pl = d.get("place") or {}
            cc = pl.get("country_code")
            u = d.get("user") or {}
            ext = d.get("extended_tweet")
            text = d.get("text", "")
            full = (ext or {}).get("full_text") or text
            truncated = bool(ext)

            # Was this post climate-related? The 2025 run only ever saw `text`,
            # so for truncated posts score both and keep the difference.
            is_climate = climate_filter.is_climate_tweet(full.lower())
            hit_on_text = (climate_filter.is_climate_tweet(text.lower())
                           if truncated else is_climate)

            # ---- denominator over every post in the file ----
            a = agg[(cc, d.get("lang"))]
            a[0] += 1
            if "retweeted_status" in d:
                a[1] += 1
            if d.get("in_reply_to_status_id") is not None:
                a[2] += 1
            if d.get("is_quote_status"):
                a[3] += 1
            if truncated:
                a[4] += 1
            if u.get("verified"):
                a[5] += 1
            if is_climate:
                a[6] += 1
            if hit_on_text:
                a[7] += 1
            if u.get("id_str"):
                hll_add(hll[cc], u["id_str"])

            # ---- the climate posts themselves ----
            if not is_climate:
                continue
            if require_place and not pl:
                continue
            try:
                iso = to_iso(d["created_at"])
            except Exception:
                continue
            if year is not None and iso[:4] != str(year):
                continue

            ents = (ext or {}).get("entities") or d.get("entities") or {}
            rows.append([
                iso, d.get("id_str"), pl.get("country"), cc, pl.get("full_name"),
                pl.get("place_type"), int(d.get("coordinates") is not None),
                d.get("lang"), strip_source(d.get("source")),
                text, full, int(truncated), int(hit_on_text),
                " ".join(h.get("text", "") for h in ents.get("hashtags", [])),
                int("retweeted_status" in d), int(bool(d.get("is_quote_status"))),
                int(d.get("in_reply_to_status_id") is not None),
                d.get("retweet_count"), d.get("favorite_count"),
                d.get("reply_count"), d.get("quote_count"),
                u.get("id_str"), u.get("screen_name"), u.get("name"), u.get("verified"),
                (u.get("description") or "").replace("\n", " ")[:400],
                (u.get("location") or "")[:150],
                u.get("followers_count"), u.get("friends_count"), u.get("statuses_count"),
                u.get("listed_count"), u.get("favourites_count"), u.get("created_at"),
                u.get("default_profile"), u.get("default_profile_image"),
            ])

    os.makedirs(outdir, exist_ok=True)

    def dump(suffix, header, body):
        with open(os.path.join(outdir, stem + suffix), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(body)

    dump(".tweets.csv", FIELDS, rows)
    dump(".agg.csv", AGG_FIELDS, [[cc, lg] + v for (cc, lg), v in agg.items()])
    with gzip.open(os.path.join(outdir, stem + ".hll.pkl.gz"), "wb") as fh:
        pickle.dump({k: bytes(v) for k, v in hll.items()}, fh, protocol=5)

    return stem, total, len(rows), bad


def run_parallel(raw_dir, outdir, workers=7, pattern="Global-EC-", year=None,
                 limit=None, progress_every=25):
    """
    Run `process` over every raw file, skipping any already recorded in done.log
    so an interrupted run can be restarted.
    """
    import time
    from concurrent.futures import ProcessPoolExecutor, as_completed

    os.makedirs(outdir, exist_ok=True)
    donelog = os.path.join(outdir, "done.log")
    done = set(open(donelog).read().split()) if os.path.exists(donelog) else set()

    files = sorted(f for f in os.listdir(raw_dir)
                   if f.startswith(pattern) and f.endswith(".json.gz"))
    todo = [f for f in files if f not in done][:limit]
    print(f"{len(files):,} raw files, {len(done):,} done, {len(todo):,} to process, "
          f"{workers} workers", flush=True)
    if not todo:
        return 0, 0

    t0, n_posts, n_climate, k = time.time(), 0, 0, 0
    with ProcessPoolExecutor(workers) as ex, open(donelog, "a") as dl:
        futs = {ex.submit(process, os.path.join(raw_dir, f), outdir, year): f
                for f in todo}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                stem, total, kept, bad = fut.result()
            except Exception as e:
                print(f"!! FAILED {f}: {e}", flush=True)
                continue
            k += 1
            n_posts += total
            n_climate += kept
            dl.write(f + "\n")
            dl.flush()
            if k % progress_every == 0 or k == len(todo):
                el = time.time() - t0
                print(f"[{k:5d}/{len(todo)}] {n_posts/1e6:7.1f}M posts, "
                      f"{n_climate:,} climate, {el/60:.1f} min, "
                      f"ETA {(len(todo)-k)*el/max(k,1)/60:.0f} min", flush=True)

    print(f"\nDone: {k} files, {n_posts:,} posts, {n_climate:,} climate posts, "
          f"{(time.time()-t0)/60:.1f} min")
    return n_posts, n_climate
