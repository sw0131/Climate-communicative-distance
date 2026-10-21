"""Masks for the robustness checks, and the loop that runs them.

Both checks are the same computation on a subset of the posts, so both go through
`aggregate.country_indices(mask=...)` and nothing here reimplements an index.

    S1  raise the posts-per-country threshold          R1.3.a
    S2  drop every account we can identify as not an individual   R1.3.c, R2.3

Two things decided while building this.

**S2 is one exclusion, not four.** Automation can be read off the posting client, which
covers the whole corpus but only catches accounts posting through a scheduler; or off the
profile by hand, which is accurate but only the 200 busiest accounts were read. Each
misses what the other catches, so the union is the answer and reporting them separately
would look like picking the flattering one.

**No exclusion by posting frequency.** R1.3.c mentions "patterns across users related to
posting frequency", but frequency does not identify the accounts it is worried about in
this corpus: the hand-labelled automated accounts have a median lifetime rate of 0.2 posts
a day against 15.0 for individuals, because the automation here is single-purpose feed
accounts rather than high-throughput spam. Cutting the busiest 1% of accounts removes 27%
of the corpus, of which 63% of the labelled part is individuals and only 6.5% automated.
`high_volume` and `high_rate` stay in this module because section 4 of the notebook uses
them to show that, but neither is applied as a robustness check.
"""

import numpy as np
import pandas as pd

import aggregate as ag
import config as cfg
import descriptives as de

# The developed-country alliances, exactly as the manuscript's Notes list them. The
# abstract's leading claim is that these show greater communicative distance than the
# developing-country alliances, so every check has to be read against this split.
#
# The unit here is a country, matching the figures and Table 1, so the EU contributes its
# 27 member states rather than one row.
#
# PROVISIONAL: the paper groups countries into seven UNFCCC alliances (G77 and China, LDC,
# SIDS, Arab Group, EU, Umbrella, EIG) and only the Umbrella and EIG memberships are
# written out in the text. Until the published notebook's lists are recovered, "developed"
# means Umbrella + EIG + the EU and "other" is everything else, which tests the direction
# of the claim but not the seven-way ordering.
UMBRELLA = ["Australia", "Canada", "Japan", "New Zealand", "Kazakhstan", "Norway",
            "Russian Federation", "Ukraine", "United States of America"]
EIG = ["Mexico", "Liechtenstein", "Monaco", "Republic of Korea", "Switzerland"]
DEVELOPED = set(UMBRELLA) | set(EIG) | set(cfg.EU_MEMBER_COUNTRIES) | {"EU"}


# Categories from the hand labelling of the top 200 accounts (notebook 08 / T5).
INSTITUTIONAL = ["organization", "government"]
NON_INDIVIDUAL = ["automated", "organization", "government"]
UNCLEAR = ["unclear"]

# `descriptives.guess_account_type` sorts accounts into finer bins than the hand
# labelling uses, so the two have to be put on the same footing before they can be
# compared. There is no bin for "unclear" -- the heuristic reads a profile and always
# commits -- so those 27 accounts are unguessable by construction and are reported
# separately rather than counted as errors.
GUESS_TO_LABEL = {
    "individual?": "individual",
    "automated": "automated",
    "jobs": "automated",
    "news": "organization",
    "ngo_research": "organization",
    "business": "organization",
    "government": "government",
}


def guess_accuracy(lab):
    """How well the automated first guess did, which is the case for labelling by hand."""
    g = lab.auto_guess.map(GUESS_TO_LABEL)
    determinable = lab.category != "unclear"
    return {
        "n": len(lab),
        "agreement": float((g == lab.category).mean()),
        "n_determinable": int(determinable.sum()),
        "agreement_determinable": float((g[determinable] == lab.category[determinable]).mean()),
        "individual_vs_not": float(((g == "individual") == (lab.category == "individual")).mean()),
    }


def read_labels(path):
    """
    The hand-labelled top-200 table, keyed on `screen_name`.

    `user_id` cannot be used as the key: Twitter snowflake ids are 19 digits and the
    round trip through a spreadsheet turns them into float64, which keeps only 15-16
    significant digits, so 81 of the 200 ids no longer identify an account. The
    screen names survive intact and match the corpus exactly -- the assertion below
    checks that, by requiring the post counts to agree row by row.
    """
    lab = pd.read_csv(path)
    lab["screen_name"] = lab.screen_name.astype(str)
    assert lab.screen_name.is_unique, "screen names are the join key and must be unique"
    return lab


def check_labels(posts, lab):
    """Post counts recovered by screen name must equal the ones in the table."""
    got = posts.screen_name.value_counts().reindex(lab.screen_name)
    bad = int((got.to_numpy() != lab.posts.to_numpy()).sum())
    return {"rows": len(lab), "mismatched": bad, "posts": int(lab.posts.sum())}


def accounts_in(lab, categories):
    return set(lab.loc[lab.category.isin(categories), "screen_name"])


def by_client(posts):
    """Automated by the app that posted. Catches the whole corpus, not just the top 200."""
    return posts.source.isin(de.AUTOMATION_SOURCES).to_numpy()


def by_label(posts, lab, categories):
    """Automated or institutional by hand. Accurate, but only the top 200 were read."""
    return posts.screen_name.isin(accounts_in(lab, categories)).to_numpy()


def non_individual(posts, lab, include_unclear=False):
    """
    Every account we can identify as something other than a private individual, by either
    method: the posting client for the whole corpus, the hand labels for the 200 busiest.

    `include_unclear` adds the 27 accounts whose profile does not settle the question,
    which gives the upper bound on how much of the corpus this could account for.
    """
    cats = NON_INDIVIDUAL + (UNCLEAR if include_unclear else [])
    return by_client(posts) | by_label(posts, lab, cats)


def conclusions(idx, ga_groups=None):
    """
    The claims each check has to be read against, rather than the index values alone.

    Aggregates cannot move further than the country values they average, so reporting the
    aggregates directly settles the figures without recomputing them:

        ADI_gap   developed minus other; the abstract says this is positive
        OSI_gap   other minus developed; the abstract says this is positive
        tvr_order groups from most under- to most over-discussed online; the text says
                  G1 is the most under-discussed and G5 the most over-discussed
    """
    dev = idx.index.isin(DEVELOPED)
    out = {
        "parties": len(idx),
        "n_developed": int(dev.sum()),
        "ADI_developed": idx.ADI[dev].mean(),
        "ADI_other": idx.ADI[~dev].mean(),
        "OSI_developed": idx.OSI[dev].mean(),
        "OSI_other": idx.OSI[~dev].mean(),
    }
    out["ADI_gap"] = out["ADI_developed"] - out["ADI_other"]
    out["OSI_gap"] = out["OSI_other"] - out["OSI_developed"]
    tvr = {g: idx[f"{g}_TVR"].replace([np.inf, -np.inf], np.nan).mean() for g in ag.GROUPS}
    order = sorted(tvr, key=tvr.get)
    out["tvr_order"] = " < ".join(order)
    out["G1_most_under"] = order[0] == "G1"
    out["G5_G6_most_over"] = set(order[-2:]) == {"G5", "G6"}
    for g in ag.GROUPS:
        out[f"{g}_TVR"] = tvr[g]
    return out


def high_volume(posts, top_frac=0.01):
    """
    The busiest accounts by number of posts in the sample. R1.3.c asks about posting
    frequency; this is the blunt version of it and cuts the most.
    """
    counts = posts.user_id.value_counts()
    k = max(1, int(round(len(counts) * top_frac)))
    return posts.user_id.isin(set(counts.index[:k])).to_numpy(), k


def high_rate(posts, per_day=50):
    """
    Lifetime posts per day, from `statuses_count` and the account's creation date. A
    different cut from `high_volume`: it catches an account that posts constantly
    about everything, not one that posted a lot about climate.
    """
    rate = de.posting_rate(posts.statuses_count, posts.user_created_at)
    return (rate > per_day).to_numpy(), int(posts.loc[rate > per_day, "user_id"].nunique())


def run(posts, recall, stance, ga_party, checks, baseline_label="all posts",
        stance_score=None):
    """
    Run a list of (label, mask, min_posts) against the baseline and collect one row
    per check. `mask` is True for posts to KEEP; None keeps everything.

    Returns (summary, indices) where `indices` maps label -> the full index frame, so
    a check can be inspected Party by Party afterwards.
    """
    base, base_meta = ag.country_indices(posts.party, recall, stance, ga_party,
                                         stance_score=stance_score, label=baseline_label)
    n = len(posts)
    out = {baseline_label: base}
    rows = [{"check": baseline_label, "posts": base_meta["posts"],
             "pct_of_posts": 100.0 * base_meta["posts"] / n, "parties": len(base),
             "ADI_mean": base.ADI.mean(), "OSI_mean": base.OSI.mean(),
             "ADI_pearson": 1.0, "ADI_spearman": 1.0, "OSI_pearson": 1.0,
             "shared_parties": len(base)}]

    for label, mask, min_posts in checks:
        idx, meta = ag.country_indices(posts.party, recall, stance, ga_party,
                                       mask=mask, min_posts=min_posts,
                                       stance_score=stance_score, label=label)
        cmp, shared = ag.compare(base, idx, "base", "sub")
        out[label] = idx
        rows.append({
            "check": label,
            "posts": meta["posts"],
            "pct_of_posts": 100.0 * meta["posts"] / n,
            "parties": len(idx),
            "ADI_mean": idx.ADI.mean(),
            "OSI_mean": idx.OSI.mean(),
            "ADI_pearson": cmp.loc["ADI", "pearson"],
            "ADI_spearman": cmp.loc["ADI", "spearman"],
            "OSI_pearson": cmp.loc["OSI", "pearson"],
            "shared_parties": shared,
        })
    return pd.DataFrame(rows).set_index("check"), out


def equal_n(posts, recall, stance, ga_party, n=200, draws=20, seed=8521,
            stance_score=None):
    """
    Draw exactly `n` posts from every Party that has at least that many, and compute
    ADI on the draw. Repeated `draws` times.

    The point is not that a Party's ADI is better estimated this way -- it is worse,
    because most of the data is thrown away. The point is that after this every Party
    rests on the same number of posts, so if ADI still lines up with the full-sample
    values then the sample imbalance R1.3.a asks about is not producing the pattern.
    """
    rng = np.random.default_rng(seed)
    counts = posts.party.value_counts()
    eligible = counts[counts >= n].index
    pos = {p: np.flatnonzero((posts.party == p).to_numpy()) for p in eligible}

    adi = {}
    for d in range(draws):
        take = np.concatenate([rng.choice(pos[p], n, replace=False) for p in eligible])
        mask = np.zeros(len(posts), dtype=bool)
        mask[take] = True
        idx, _ = ag.country_indices(posts.party, recall, stance, ga_party,
                                    mask=mask, min_posts=0, label=f"equal-n draw {d}")
        adi[d] = idx.ADI
    return pd.DataFrame(adi), list(eligible)
