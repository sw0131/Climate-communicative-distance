"""Every number the Results sections report, computed the same way for several versions
of the corpus.

The point of the file is comparison. The 2025 processing read the 140-character `text`
field instead of `extended_tweet.full_text`, which both shortened what each post said and
changed which posts passed the climate filter at all. Three versions separate those:

    truncated     the published per-post results, read at 140 characters
    full text     the rebuild, reading the whole post          <- the revision
    untruncated   the rebuild restricted to posts that were never truncated, where the
                  two readings are the same text, so only sample composition differs

Sections follow the manuscript:

    2.1  topic shares, 38 topics and 6 groups, NDC side and online side
    2.2  country rankings on ADI and OSI, and the alliance aggregates
    2.3  group-level TVR and OSI with intervals, and the 38-topic quadrants
    2.4  Table 1
"""

import numpy as np
import pandas as pd

import aggregate as ag
import config as cfg
import sensitivity as sn

GROUPS = ag.GROUPS
QUADRANTS = {(True, True): "I amplified support",
             (True, False): "II amplified dissent",
             (False, True): "III silent support",
             (False, False): "IV silent dissent"}


def ndc_shares(ga_chunks):
    """
    Share of the NDC corpus on each topic and group, pooled over all chunks.

    Reported against the T1-T38 labels rather than BERTopic's own numbering, so the two
    sides of the Sankey can be read off the same rows. Figure 1 states 27.3% for G3,
    24.7% for G2 and 24.5% for G1, which is what this should reproduce.
    """
    sub = cfg.ndc_columns()
    w = ga_chunks[sub].mean()
    w = w / w.sum()
    rows = []
    for i in range(1, 39):
        j = cfg.tw_topic_to_bertopic_index(i)
        rows.append({"topic": f"T{i}", "ndc_share": float(w[f"Subtopic_{j}"])})
    t = pd.DataFrame(rows).set_index("topic")
    g = pd.Series({g: sum(float(w[f"Subtopic_{i}"]) for i in ids)
                   for g, ids in cfg.TOPIC_GROUPS_NDC.items()}, name="ndc_share")
    return t, g


def online_shares(recall):
    """
    Share of the online corpus on each topic and group: total recall on a topic over
    total recall on all 38, pooled over every post in the version.

    Figure 1 is drawn on the whole corpus, not on the countries that clear the posting
    threshold, so this deliberately does not filter by country.
    """
    tc = cfg.tw_columns()
    s = recall[tc].sum()
    s = s / s.sum()
    t = pd.DataFrame({"online_share": s.to_numpy()},
                     index=[c.replace("Topic_", "T") for c in tc])
    t.index.name = "topic"
    g = pd.Series({g: float(s[[f"Topic_{i}" for i in ids]].sum())
                   for g, ids in cfg.TOPIC_GROUPS_TW.items()}, name="online_share")
    return t, g


def country_ranking(idx):
    """Countries ordered by ADI, with their OSI rank alongside."""
    out = idx[["n_posts", "ADI", "OSI"]].copy()
    out["ADI_rank"] = out.ADI.rank(ascending=False).astype(int)
    out["OSI_rank"] = out.OSI.rank(ascending=False).astype(int)
    return out.sort_values("ADI", ascending=False)


def alliance(idx):
    """
    ADI and OSI by UNFCCC alliance, for the right-hand panel of Figure 2.

    Only the Umbrella and EIG memberships are written out in the manuscript, so the
    developing-country alliances are not separable here and fall into "other".
    """
    grp = pd.Series("other", index=idx.index)
    grp[idx.index.isin(cfg.EU_MEMBER_COUNTRIES)] = "EU"
    grp[idx.index.isin(sn.UMBRELLA)] = "Umbrella"
    grp[idx.index.isin(sn.EIG)] = "EIG"
    t = idx.groupby(grp).agg(countries=("ADI", "size"), posts=("n_posts", "sum"),
                             ADI=("ADI", "mean"), OSI=("OSI", "mean"))
    t["ADI_vs_mean"] = t.ADI - idx.ADI.mean()
    t["OSI_vs_mean"] = t.OSI - idx.OSI.mean()
    return t.sort_values("ADI", ascending=False)


def _mean_ci(v):
    v = v.replace([np.inf, -np.inf], np.nan).dropna()
    return {"n": len(v), "mean": v.mean(), "sd": v.std(),
            "ci_low": v.mean() - 1.96 * v.sem(), "ci_high": v.mean() + 1.96 * v.sem()}


def group_table(idx):
    """Group-level TVR and OSI, each with SD and a 95% interval (R2.7)."""
    rows = []
    for g in GROUPS:
        tvr, osi = _mean_ci(idx[f"{g}_TVR"]), _mean_ci(idx[f"{g}_OSI"])
        rows.append({"group": g, "name": cfg.GROUP_NAMES[g], "n_countries": tvr["n"],
                     **{f"TVR_{k}": v for k, v in tvr.items() if k != "n"},
                     **{f"OSI_{k}": v for k, v in osi.items() if k != "n"}})
    return pd.DataFrame(rows).set_index("group")


def topic_quadrants(idx, topic_table=None):
    """
    The 38 topics placed in Figure 4's quadrants: TVR above or below zero on one axis,
    OSI above or below its mean across topics on the other.

    The OSI split is a comparison against a mean, so it survives any monotone rescaling
    of OSI -- which matters, because the scale the published figure used is not recovered
    (its caption gives a global mean of 0.73, which nothing here produces).
    """
    rows = []
    for i in range(1, 39):
        tvr = idx[f"T{i}_TVR"].replace([np.inf, -np.inf], np.nan).dropna()
        osi = idx[f"T{i}_OSI"].replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({"topic": f"T{i}", "n_countries": len(tvr),
                     "TVR_mean": tvr.mean(), "TVR_sd": tvr.std(),
                     "OSI_mean": osi.mean(), "OSI_sd": osi.std()})
    t = pd.DataFrame(rows).set_index("topic")
    cut = t.OSI_mean.mean()
    t["over_discussed"] = t.TVR_mean > 0
    t["supportive"] = t.OSI_mean > cut
    t["quadrant"] = [QUADRANTS[(o, s)] for o, s in zip(t.over_discussed, t.supportive)]
    t.attrs["osi_cut"] = float(cut)
    if topic_table is not None:
        t["group"] = topic_table.reindex(t.index)["group"]
        t["label"] = (topic_table.reindex(t.index)["sentence"]
                      .str.split(" - ").str[0].str[4:])
    return t


def compare_versions(tables, key, columns):
    """Put the same table from several versions side by side on a shared index."""
    out = {}
    for name, t in tables.items():
        out[name] = t[columns] if isinstance(columns, list) else t[columns]
    return pd.concat(out, axis=1)
