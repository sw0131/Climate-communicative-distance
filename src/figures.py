"""The four main figures.

Each function returns a matplotlib Figure and draws nothing else, so the notebook decides
where things are saved and at what size.

Two things are carried over from the 2025 figures rather than reinvented:

* **Figure 1 is drawn with `sankeyflow`**, the package the original used, one panel per
  side. The original rendered it six times, once per thematic group with that group's
  ribbons in black and the rest in grey, and assembled the pieces in Photoshop; that is
  reproduced by `sankey_panel(highlight=...)`, while `sankey_panel()` with no highlight
  draws the plain coloured version;
* **Figure 2's ADI axis is the rescaled index.** ADI is min-max scaled to [0, 1] before
  plotting, which is what the published legend's bands (<0.15, 0.15-0.25, 0.25-0.55,
  >=0.55) refer to; the raw ADI is a Jensen-Shannon divergence about a third that size.
  OSI is already on the stance scale and is plotted as it is.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

import config as cfg

_L = json.load(open(os.path.join(cfg.ASSETS_DIR, "figure_labels.json"), encoding="utf-8"))
TOPIC_LABELS = _L["TOPIC_LABELS"]
GROUP_LABELS = _L["GROUP_LABELS"]
GROUP_COLORS = _L["GROUP_COLORS"]
GROUPS = [f"G{i}" for i in range(1, 7)]

# The seven UNFCCC negotiation groups, exactly as the 2025 notebook lists them.
ALLIANCES = {
    "G77 & China": ["Argentina", "Bahamas", "Bahrain", "Bangladesh", "Barbados",
        "Botswana", "Brazil", "Cameroon", "Chile", "China", "Colombia", "Costa Rica",
        "Democratic Republic of the Congo", "Ecuador", "Egypt", "Ethiopia", "Gambia",
        "Ghana", "India", "Indonesia", "Iraq", "Jamaica", "Jordan", "Kenya", "Lebanon",
        "Lesotho", "Malawi", "Malaysia", "Maldives", "Mozambique", "Namibia", "Nepal",
        "Nigeria", "Oman", "Pakistan", "Philippines", "Qatar", "Rwanda", "Saudi Arabia",
        "Sierra Leone", "Singapore", "Somalia", "South Africa", "Sri Lanka", "Thailand",
        "Togo", "Trinidad and Tobago", "Tunisia", "Uganda", "United Arab Emirates",
        "United Republic of Tanzania", "Venezuela", "Vietnam", "Zambia", "Zimbabwe"],
    "SIDS": ["Bahamas", "Barbados", "Jamaica", "Maldives", "Philippines", "Singapore",
        "Trinidad and Tobago"],
    "LDC": ["Bangladesh", "Democratic Republic of the Congo", "Ethiopia", "Gambia",
        "Lesotho", "Malawi", "Mozambique", "Nepal", "Rwanda", "Sierra Leone", "Somalia",
        "Togo", "Uganda", "United Republic of Tanzania", "Zambia"],
    "Arab Group": ["Bahrain", "Egypt", "Iraq", "Jordan", "Lebanon", "Oman", "Qatar",
        "Saudi Arabia", "Somalia", "Tunisia", "United Arab Emirates"],
    "Umbrella Group": ["Australia", "Canada", "Japan", "New Zealand", "Kazakhstan",
        "Norway", "Russian Federation", "Ukraine", "United States of America"],
    "EIG": ["Mexico", "Republic of Korea", "Switzerland"],
    "EU": list(cfg.EU_MEMBER_COUNTRIES),
}

# Published legend bands. ADI's apply to the min-max rescaled index.
ADI_BANDS = [(0.15, "low", "<0.15"), (0.25, "medium-low", "0.15-0.25"),
             (0.55, "medium-high", "0.25-0.55"), (np.inf, "high", ">=0.55")]
ADI_SIZES = [40, 90, 170, 300]
OSI_CMAP = LinearSegmentedColormap.from_list(
    "osi", ["#C2277E", "#E2A0B4", "#EBD9C4", "#C5D6A0", "#5E9B54"])


def rescale(v):
    """Min-max to [0, 1], the axis the published figures use for ADI."""
    v = np.asarray(v, dtype=float)
    lo, hi = np.nanmin(v), np.nanmax(v)
    return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)


# --- Figure 1 -------------------------------------------------------------------

def sankey_flows(shares, side):
    """
    (source, target, value) triples for one side of Figure 1.

    side="ndc"      topics flow into groups, so the topic is the source
    side="online"   groups flow out into topics, mirroring the panel
    """
    flows = []
    for g, ids in cfg.TOPIC_GROUPS_TW.items():
        gname = f"{g}: {GROUP_LABELS[g]}"
        for i in ids:
            t = f"T{i}"
            tname = f"{t}: {TOPIC_LABELS[t]}"
            v = float(shares[t])
            if v <= 0:
                continue
            flows.append((tname, gname, v) if side == "ndc" else (gname, tname, v))
    return flows


def _node_order(flows, side):
    w = {}
    for a, b, v in flows:
        w[a] = w.get(a, 0) + v
        w[b] = w.get(b, 0) + v
    topics = [f"T{i}: {TOPIC_LABELS[f'T{i}']}" for i in range(1, 39)]
    groups = [f"{g}: {GROUP_LABELS[g]}" for g in GROUPS]
    cols = [topics, groups] if side == "ndc" else [groups, topics]
    return [[(n, w[n]) for n in col if n in w] for col in cols]


def sankey_panel(shares, side, highlight=None, figsize=(12, 8), dpi=200,
                 labels=False, dim="#D9D9D9", dim_alpha=0.25):
    """
    One side of Figure 1. `highlight` is a group key such as "G3"; its ribbons are drawn
    last and in black while everything else is dimmed, which is how the published figure
    was built up group by group.
    """
    from sankeyflow import Sankey

    flows = sankey_flows(shares, side)
    hl = f"{highlight}: {GROUP_LABELS[highlight]}" if highlight else None
    plain, bright = [], []
    for a, b, v in flows:
        node = b if side == "ndc" else a
        if hl is None:
            plain.append((a, b, v, {"color": GROUP_COLORS[node.split(":")[0]],
                                    "alpha": 0.85}))
        elif node == hl:
            bright.append((a, b, v, {"color": "black"}))
        else:
            plain.append((a, b, v, {"color": dim, "alpha": dim_alpha}))

    fig = plt.figure(figsize=figsize, dpi=dpi)
    s = Sankey(flows=plain + bright, nodes=_node_order(flows, side),
               node_opts=dict(visible=False))
    s.draw()
    for col in s.nodes:
        for n in col:
            if not labels:
                n.text.set_text("")
            # sankeyflow colours the node bars from a cycle; the published figure uses
            # one dark bar per node, so recolour whatever patch it created.
            for attr in ("patch", "rect", "artist"):
                p = getattr(n, attr, None)
                if p is not None and hasattr(p, "set_facecolor"):
                    p.set_facecolor("#3A3A3A")
                    p.set_edgecolor("none")
    plt.tight_layout()
    return fig


# --- Figure 2 -------------------------------------------------------------------

def _adi_band(v):
    for i, (hi, _, _) in enumerate(ADI_BANDS):
        if v < hi:
            return i
    return len(ADI_BANDS) - 1


def bubble_map(idx, world, label_countries=(), figsize=(20, 10), dpi=200):
    """
    Figure 2 left: one bubble per country, size from the rescaled ADI, colour from OSI.
    `world` is a GeoDataFrame whose `name` column carries the shapefile country names.
    """
    d = idx.copy()
    d["ADI_scaled"] = rescale(d.ADI)
    d["size"] = [ADI_SIZES[_adi_band(v)] for v in d.ADI_scaled]

    w = world[world.name.isin(d.index)].copy()
    pts = w.geometry.representative_point()
    w["lon"], w["lat"] = pts.x.to_numpy(), pts.y.to_numpy()
    w = w.drop_duplicates("name").set_index("name")
    common = [c for c in d.index if c in w.index]
    d = d.loc[common]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_facecolor("#F7F9FB")
    world.plot(ax=ax, edgecolor="#BFBFBF", facecolor="#FCFCFC", linewidth=0.3)
    norm = TwoSlopeNorm(vmin=float(d.OSI.min()), vcenter=float(d.OSI.mean()),
                        vmax=float(d.OSI.max()))
    ax.scatter(w.loc[common, "lon"], w.loc[common, "lat"], s=d["size"],
               c=d.OSI, cmap=OSI_CMAP, norm=norm,
               edgecolor="black", linewidth=0.3, alpha=0.88, zorder=5)
    for c in label_countries:
        if c in common:
            ax.annotate(c, (w.loc[c, "lon"], w.loc[c, "lat"]), fontsize=9,
                        fontweight="bold", xytext=(6, 6), textcoords="offset points")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.axis("off")
    ax.text(0.005, 0.99, "a", transform=ax.transAxes, ha="left", va="top",
            fontsize=15, fontweight="bold")

    handles = [Line2D([], [], marker="o", ls="", markersize=np.sqrt(s) * 0.8,
                      markerfacecolor="#BBBBBB", markeredgecolor="black",
                      label=f"{name} {rng}")
               for s, (_, name, rng) in zip(ADI_SIZES, ADI_BANDS)]
    ax.legend(handles=handles, loc="lower left", frameon=False,
              title="Attention Distance Index (ADI)", labelspacing=1.1)
    sm = plt.cm.ScalarMappable(cmap=OSI_CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.03, pad=0.02)
    cb.set_label("Online Stance Index (OSI)")
    plt.tight_layout()
    return fig


def alliance_frame(idx):
    """One row per country per alliance it belongs to, for the Figure 2 boxplots."""
    rows = []
    scaled = pd.Series(rescale(idx.ADI), index=idx.index)
    for name, members in ALLIANCES.items():
        for c in members:
            if c in idx.index:
                rows.append({"alliance": name, "country": c,
                             "ADI": scaled[c], "OSI": idx.OSI[c]})
    return pd.DataFrame(rows)


def alliance_boxplots(idx, figsize=(7, 9), dpi=200):
    """Figure 2 right: ADI over the seven alliances, then OSI, each with the total mean."""
    af = alliance_frame(idx)
    order = list(ALLIANCES)
    scaled = pd.Series(rescale(idx.ADI), index=idx.index)
    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
    for ax, col, title, total in [
            (axes[0], "ADI", "(b) Attention Distance Index", scaled.mean()),
            (axes[1], "OSI", "(c) Online Stance Index", idx.OSI.mean())]:
        data = [af.loc[af.alliance == a, col].dropna().to_numpy() for a in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, markerfacecolor="none"))
        for patch, c in zip(bp["boxes"], plt.cm.tab20.colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        for med in bp["medians"]:
            med.set_color("#444444")
        ax.axhline(total, color="#CC3333", ls="--", lw=1.2, label="Total mean")
        ax.set_xticks(range(1, len(order) + 1))
        # the n is required in the legend of a panel showing a distribution, so it is
        # printed on the axis as well
        n = {a: int((af.alliance == a).sum()) for a in order}
        ax.set_xticklabels([f"{a}\n(n = {n[a]})" for a in order],
                           rotation=90, fontsize=9, fontweight="bold")
        ax.set_ylabel(col)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    return fig


# --- Figure 3 -------------------------------------------------------------------

def group_panels(idx, figsize=(16, 9), dpi=200):
    """
    Figure 3: one panel per thematic group. Countries are ordered by their TVR on that
    group and coloured by their OSI on it, with the group's means in the corner.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=dpi)
    osi_all = idx[[f"{g}_OSI" for g in GROUPS]].to_numpy().ravel()
    norm = TwoSlopeNorm(vmin=float(np.nanpercentile(osi_all, 2)),
                        vcenter=float(np.nanmean(osi_all)),
                        vmax=float(np.nanpercentile(osi_all, 98)))
    for ax, g in zip(axes.ravel(), GROUPS):
        tvr = idx[f"{g}_TVR"].replace([np.inf, -np.inf], np.nan)
        d = pd.DataFrame({"tvr": tvr, "osi": idx[f"{g}_OSI"]}).dropna().sort_values("tvr")
        ax.axhspan(0, max(d.tvr.max(), 0.01), color="#F6C9A8", alpha=0.35, zorder=0)
        ax.axhspan(min(d.tvr.min(), -0.01), 0, color="#BBD6E8", alpha=0.35, zorder=0)
        ax.scatter(range(len(d)), d.tvr, c=d.osi, cmap=OSI_CMAP, norm=norm,
                   s=26, edgecolor="black", linewidth=0.25, zorder=3)
        ax.axhline(0, color="black", lw=0.8, ls="--", zorder=2)
        ax.set_title(f"{g}: {cfg.GROUP_NAMES[g]}", fontsize=10.5)
        ax.set_xticks([])
        ax.set_xlabel(f"{len(d)} countries", fontsize=8)
        ax.text(0.02, 0.04, f"TVR {d.tvr.mean():+.2f}\nOSI {d.osi.mean():.2f}",
                transform=ax.transAxes, fontsize=8.5, va="bottom")
    fig.supylabel("TVR   (orange: over-discussed online, blue: under-discussed)",
                  fontsize=10)
    sm = plt.cm.ScalarMappable(cmap=OSI_CMAP, norm=norm)
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.04, pad=0.06)
    cb.set_label("Topic-level OSI")
    return fig


# --- Figure 4 -------------------------------------------------------------------

def quadrant_plot(topics, figsize=(11, 9), dpi=200, annotate=True):
    """
    Figure 4: the 38 topics on mean TVR against mean OSI, split at TVR = 0 and the mean
    of OSI across topics. `topics` is the frame `results.topic_quadrants` returns.
    """
    cut = float(topics.OSI_mean.mean())
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axvline(0, color="#777777", ls="--", lw=1)
    ax.axhline(cut, color="#777777", ls="--", lw=1)
    for g in GROUPS:
        sub = topics[topics.group == g] if "group" in topics.columns else topics.iloc[0:0]
        ax.scatter(sub.TVR_mean, sub.OSI_mean, s=90, color=GROUP_COLORS[g],
                   edgecolor="black", linewidth=0.4, label=f"{g}: {cfg.GROUP_NAMES[g]}",
                   zorder=3)
    if annotate:
        for t, r in topics.iterrows():
            ax.annotate(t, (r.TVR_mean, r.OSI_mean), fontsize=7.5,
                        xytext=(5, 4), textcoords="offset points")
    xr = topics.TVR_mean.max() - topics.TVR_mean.min()
    yr = topics.OSI_mean.max() - topics.OSI_mean.min()
    for (dx, dy), lab in [((0.98, 0.98), "I  amplified support"),
                          ((0.98, 0.02), "II  amplified dissent"),
                          ((0.02, 0.98), "III  silent support"),
                          ((0.02, 0.02), "IV  silent dissent")]:
        ax.text(dx, dy, lab, transform=ax.transAxes, fontsize=10, color="#555555",
                ha="right" if dx > 0.5 else "left", va="top" if dy > 0.5 else "bottom")
    ax.set_xlabel("TVR   (negative: under-discussed online relative to NDCs)")
    ax.set_ylabel("OSI   (topic-level stance)")
    ax.set_xlim(topics.TVR_mean.min() - 0.08 * xr, topics.TVR_mean.max() + 0.12 * xr)
    ax.set_ylim(topics.OSI_mean.min() - 0.10 * yr, topics.OSI_mean.max() + 0.10 * yr)
    ax.legend(frameon=False, fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    plt.tight_layout()
    return fig


# --- Supplementary: how the corpus is spread across countries ------------------

# The shapefile spellings are too long for an axis. Only the unwieldy ones need a
# short form; everything else is already short enough to print as it is.
SHORT_NAME = {
    "United States of America": "United States",
    "U.K. of Great Britain and Northern Ireland": "United Kingdom",
    "United Republic of Tanzania": "Tanzania",
    "Democratic Republic of the Congo": "DR Congo",
    "Republic of Korea": "South Korea",
    "Russian Federation": "Russia",
    "Iran (Islamic Republic of)": "Iran",
    "Venezuela (Bolivarian Republic of)": "Venezuela",
    "Bolivia (Plurinational State of)": "Bolivia",
    "Moldova, Republic of": "Moldova",
    "The former Yugoslav Republic of Macedonia": "North Macedonia",
    "Lao People's Democratic Republic": "Laos",
    "Syrian Arab Republic": "Syria",
    "Trinidad and Tobago": "Trinidad & Tobago",
    "Bosnia and Herzegovina": "Bosnia & Herz.",
}


def short(name):
    return SHORT_NAME.get(name, name)


def country_coverage(by_country, top_n=20, figsize=(14, 7), dpi=200):
    """
    Two panels answering Referee 1's first and third sentences at once.

    (a) the countries that contribute most, with the remainder collapsed into one bar.
        The axis is logarithmic because the range spans three orders of magnitude,
        which is itself the point.
    (b) the share of a country's own posts that are climate related, against how many
        posts that country contributes. A country can be large on the left panel
        because the platform is widely used there rather than because climate is
        discussed more, and this panel separates the two.
    """
    d = by_country[by_country.in_analysis].sort_values("posts", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi,
                             gridspec_kw={"width_ratios": [1, 1.15]})

    # (a) top_n countries plus the remainder
    top = d.head(top_n)
    rest = d.iloc[top_n:]
    names = [short(c) for c in top.index] + [f"Other {len(rest)} countries"]
    vals = list(top.posts) + [int(rest.posts.sum())]
    colors = ["#7FA8C9"] * top_n + ["#C4C4C4"]
    y = np.arange(len(vals))[::-1]

    ax = axes[0]
    ax.barh(y, vals, color=colors, edgecolor="none", height=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xscale("log")
    ax.set_xlabel("climate posts (log scale)")
    ax.set_title(f"(a) Posts per country, {len(d)} countries",
                 fontsize=11, fontweight="bold")
    for yy, v in zip(y, vals):
        ax.text(v * 1.12, yy, f"{v:,}", va="center", fontsize=7.5, color="#444444")
    ax.set_xlim(right=max(vals) * 3)
    cum = d.posts.cumsum() / d.posts.sum()
    ax.text(0.98, 0.02,
            f"top 5: {100 * cum.iloc[4]:.0f}% of posts\ntop {top_n}: "
            f"{100 * cum.iloc[top_n - 1]:.0f}%",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            color="#B03030")

    # (b) volume against attention
    ax = axes[1]
    share = 100 * d.climate_share_of_country
    ax.scatter(d.posts, share, s=34, color="#7FA8C9", edgecolor="black", linewidth=0.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("climate posts contributed (log scale)")
    ax.set_ylabel("climate share of that country's posts, % (log scale)")
    ax.set_title("(b) Climate share of each country's posts",
                 fontsize=11, fontweight="bold")
    for c in list(d.index[:5]) + list(share.nlargest(3).index):
        if c in d.index:
            ax.annotate(short(c), (d.posts[c], share[c]), fontsize=8,
                        xytext=(6, 4), textcoords="offset points")
    plt.tight_layout()
    return fig
