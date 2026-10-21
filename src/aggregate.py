"""Country-level indices: ADI, OSI and TVR.

Takes the per-post topic recall and topic-weighted stance and aggregates them to
Parties, then compares each Party's online topic profile with its NDC's.

    GA_cj   NDC weight on topic j, the mean over a Party's 500-word chunks
    PP_cj   online weight on topic j, the summed recall over a Party's posts,
            normalised so the 38 weights sum to one
    PS_cj   mean topic-weighted stance over a Party's posts

    ADI  Jensen-Shannon divergence between GA and PP, over the six thematic groups
    OSI  max(0, mean stance over a unit's posts), and the same measure resolved to one
         group or one topic by weighting each post's stance by its recall on it
    TVR  log(PP_ij / GA_ij), at the level of individual topics and of the six groups

Two things the published code got wrong are fixed here:

* **OSI is the plain mean stance over a unit's posts.** Equations (3), (6) and (8) of
  the manuscript describe a stance weighted by recall, and that is what an earlier version
  of this module computed, but it is not what the paper reports: weighting by recall
  divides every value by the mean recall, leaving OSI around 0.009 where the figures run
  from 0.24 to 0.86. Aggregated this way the seven UNFCCC alliances come out in exactly
  the order Figure 2 plots them (Spearman 1.000 against 0.643 for the weighted version),
  and Table 1's OSI row reproduces. The equations are what needs correcting;
* the topic-level TVR needs the T1-T38 to BERTopic renumbering
  (`cfg.tw_topic_to_bertopic_index`). The 2025 notebook applied an extra swap of
  T2 with T35 and T6 with T38 on top of it, which the topic keywords contradict;
* ADI is over the six groups, which the manuscript's Eq. (7) should say.

Everything runs through `country_indices`, which takes a boolean mask, so the
sensitivity analyses are the same computation on a subset rather than a second
implementation of it.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

import config as cfg

GROUPS = ["G1", "G2", "G3", "G4", "G5", "G6"]


def add_party(df, country_col="country"):
    """
    Place name -> English -> shapefile spelling -> Party, with the 27 EU member
    states collapsed into a single "EU" because they file one joint NDC.
    """
    out = df.copy()
    out["en_country"] = out[country_col].map(cfg.MAPPING_TW_TO_EN)
    out["shp_country"] = out.en_country.map(cfg.MAPPING_EN_TO_SHP)
    out["party"] = out.shp_country.where(
        ~out.shp_country.isin(cfg.EU_MEMBER_COUNTRIES), "EU")
    return out


def ndc_by_party(ga_chunks):
    """GA: mean topic weights per Party, plus the six group sums."""
    ga = ga_chunks.copy()
    ga["shp_country"] = ga.Country.map(cfg.MAPPING_NDC_TO_SHP)
    ga["party"] = ga.shp_country.where(
        ~ga.shp_country.isin(cfg.EU_MEMBER_COUNTRIES), "EU")
    sub = cfg.ndc_columns()
    out = ga.groupby("party", as_index=False)[sub].mean()
    for g, ids in cfg.TOPIC_GROUPS_NDC.items():
        out[g] = out[[f"Subtopic_{i}" for i in ids]].sum(axis=1)
    return out


def jsd(p, q, base=2):
    """Jensen-Shannon divergence. scipy returns the distance, so square it."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    return float(jensenshannon(p, q, base=base) ** 2)


def country_indices(party, recall, stance, ga_party, mask=None,
                    min_posts=None, label="", stance_score=None):
    """
    Aggregate to Parties and compute the indices.

    party     one row per post: the unit it belongs to, aligned with `recall`
    recall    Topic_1..Topic_38, one row per post
    stance    Topic_1_stance..Topic_38_stance, one row per post
    stance_score  the post's own stance, +1/0/-1, one row per post. Taken from a
              `stance_score` column of `stance` when not passed separately.
    ga_party  output of ndc_by_party
    mask      boolean over posts; None uses all of them
    min_posts Parties with at most this many posts are dropped
              (default cfg.MIN_TWEETS_PER_PARTY)

    Returns (indices, meta). `indices` is indexed by Party.
    """
    min_posts = cfg.MIN_TWEETS_PER_PARTY if min_posts is None else min_posts
    tc = cfg.tw_columns()
    sc = [f"{c}_stance" for c in tc]

    keep = np.ones(len(party), dtype=bool) if mask is None else np.asarray(mask, bool)
    p = party[keep]
    r = recall.loc[keep, tc]
    s = stance.loc[keep, sc]

    counts = p.value_counts()
    parties = counts[counts > min_posts].index
    sel = p.isin(parties).to_numpy()
    p, r, s = p[sel], r[sel], s[sel]

    # PP: sum the recall over a Party's posts, then normalise to a distribution
    pp = r.groupby(p.to_numpy()).sum()
    pp = pp.div(pp.sum(axis=1), axis=0)
    for g, ids in cfg.TOPIC_GROUPS_TW.items():
        pp[g] = pp[[f"Topic_{i}" for i in ids]].sum(axis=1)

    # PS: the topic-weighted stance, summed rather than averaged, because every OSI
    # below is a stance averaged with recall as the weight and so needs both sums.
    ps_sum = s.groupby(p.to_numpy()).sum()
    r_sum = r.groupby(p.to_numpy()).sum()

    if stance_score is None:
        if "stance_score" not in stance.columns:
            raise ValueError("pass stance_score, or give `stance` a stance_score column")
        stance_score = stance["stance_score"]
    o = pd.Series(np.asarray(stance_score, dtype=float))[keep].to_numpy()[sel]
    osi_unit = pd.Series(o).groupby(p.to_numpy()).mean()

    ga = ga_party.set_index("party")
    common = pp.index.intersection(ga.index)
    pp, ga = pp.loc[common], ga.loc[common]
    ps_sum, r_sum = ps_sum.loc[common], r_sum.loc[common]
    osi_unit = osi_unit.loc[common]

    out = pd.DataFrame(index=common)
    out.index.name = "party"
    out["n_posts"] = counts.reindex(common).to_numpy()
    out["ADI"] = [jsd(ga.loc[c, GROUPS].to_numpy(), pp.loc[c, GROUPS].to_numpy())
                  for c in common]
    out["OSI"] = np.maximum(0.0, osi_unit.to_numpy())

    # Group-level TVR
    for g in GROUPS:
        out[f"{g}_TVR"] = np.log(pp[g].to_numpy() / ga[g].to_numpy())

    # Group- and topic-level OSI: the same stance measure resolved to one group or one
    # topic, which is a mean of the posts' stance weighted by their recall on it. Dividing
    # by the recall keeps every OSI on the stance scale, so the group and topic values are
    # comparable with the overall one rather than being smaller by a factor of the recall.
    with np.errstate(divide="ignore", invalid="ignore"):
        for g, ids in cfg.TOPIC_GROUPS_TW.items():
            num = ps_sum[[f"Topic_{i}_stance" for i in ids]].sum(axis=1).to_numpy()
            den = r_sum[[f"Topic_{i}" for i in ids]].sum(axis=1).to_numpy()
            out[f"{g}_OSI"] = np.maximum(0.0, num / den)
        for i in range(1, 39):
            out[f"T{i}_OSI"] = np.maximum(
                0.0, ps_sum[f"Topic_{i}_stance"].to_numpy() / r_sum[f"Topic_{i}"].to_numpy())

    # Topic-level TVR. T{i} on the post side corresponds to Subtopic_{perm(i)} on
    # the NDC side; without the permutation every topic lands on the wrong one.
    for i in range(1, 39):
        j = cfg.tw_topic_to_bertopic_index(i)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"T{i}_TVR"] = np.log(pp[f"Topic_{i}"].to_numpy()
                                      / ga[f"Subtopic_{j}"].to_numpy())

    meta = {
        "label": label,
        "posts": int(sel.sum()),
        "parties_above_threshold": int(len(parties)),
        "parties_with_ndc": int(len(common)),
        "countries": None,
    }
    return out, meta


def compare(a, b, name_a="all", name_b="subset"):
    """Correlate two runs of country_indices on their shared Parties."""
    rows = []
    shared = a.index.intersection(b.index)
    for col in ["ADI", "OSI"] + [f"{g}_TVR" for g in GROUPS]:
        x, y = a.loc[shared, col], b.loc[shared, col]
        ok = np.isfinite(x) & np.isfinite(y)
        rows.append({
            "index": col,
            f"mean_{name_a}": float(x[ok].mean()),
            f"mean_{name_b}": float(y[ok].mean()),
            "pearson": float(x[ok].corr(y[ok])),
            "spearman": float(x[ok].corr(y[ok], method="spearman")),
            "mean_abs_change": float((y[ok] - x[ok]).abs().mean()),
        })
    return pd.DataFrame(rows).set_index("index"), len(shared)
