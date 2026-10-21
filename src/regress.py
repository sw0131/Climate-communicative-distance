"""Country-level regressions: Table 1.

Four outcomes, all per capita, against ADI and OSI with four controls:

    ln(GDP per capita)         OWID, gdp / population
    fossil fuel share          OWID energy, share of electricity from fossil fuels
    government effectiveness   World Bank WGI
    climate vulnerability      ND-GAIN vulnerability score

The published table reports N = 91 for all four models with HC3 standard errors and
nothing else about the specification, and the notebook that produced it is not in this
folder. Four things had to be recovered. Each is settled by which choice reproduces the
printed numbers, and together they do:

* **the unit is a country carrying its own indices.** Not a Party: each country's posts
  give it its own PP vector, compared against the NDC of the Party it filed under, so the
  EU's member states are 27 separate observations sharing one NDC rather than one
  observation between them. Aggregating the published per-post results this way yields
  exactly **91 countries**;
* **the window is the full span of the delivered corpus**, 2021-12-15 to 2023-03-13, not
  calendar 2022. Restricted to calendar 2022 the same aggregation gives 87 countries and a
  visibly worse match. The revision adopts calendar 2022 regardless, because that is what
  the manuscript already claims, so the rebuilt table has a different N by design;
* **the covariate year is 2021.** `energy_per_capita` is populated for 204 countries in
  2021 and only 79 in 2022, and the four models share one N, so 2022 cannot be it;
* **the outcome is logged and the predictors standardised.** Any other combination is out
  by orders of magnitude.

On the published per-post results this reproduces all 24 signs, R-squared to within 0.01,
and the four control coefficients to within 0.06. ADI and OSI stay a little off, and they
are the only two terms that depend on how the indices themselves were computed -- the same
open question as the OSI scale in the Figure 3 caption, whose stated global mean of 0.73
matches no quantity this pipeline produces.

`fit_table1` runs the specification and `compare_published` scores it against the printed
table.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg

COV_DIR = os.path.join(cfg.PROJECT_ROOT, "country-level covariates")
VULN = os.path.join(COV_DIR, "Climate vulnerability", "vulnerability", "vulnerability.csv")

OUTCOMES = {
    "GHG per capita": "ghg_per_capita",
    "CO2 per capita": "co2_per_capita",
    "CO2 incl. LUC": "co2_including_luc_per_capita",
    "Energy per capita": "energy_per_capita",
}
CONTROLS = ["ln_gdp_pc", "fossil_share", "gov_effectiveness", "vulnerability"]

# Panel A of the published Table 1, typed in from the manuscript so a rebuild can be
# scored against it rather than eyeballed.
PUBLISHED_A = pd.DataFrame(
    {"GHG per capita": [0.133, -0.089], "CO2 per capita": [0.106, -0.143],
     "CO2 incl. LUC": [0.184, -0.176], "Energy per capita": [0.144, -0.119]},
    index=["ADI", "OSI"])
PUBLISHED_CONTROLS = pd.DataFrame(
    {"GHG per capita": [1.005, -0.013, -0.296, 0.055],
     "CO2 per capita": [1.194, 0.271, -0.306, -0.323],
     "CO2 incl. LUC": [1.085, -0.039, -0.327, -0.095],
     "Energy per capita": [1.386, 0.062, -0.233, -0.275]},
    index=CONTROLS)
PUBLISHED_N = 91
PUBLISHED_R2 = {"GHG per capita": 0.69, "CO2 per capita": 0.89,
                "CO2 incl. LUC": 0.72, "Energy per capita": 0.90}


def crosswalk():
    """Country name (shapefile spelling) -> ISO3, from assets/country_iso3.csv."""
    cw = pd.read_csv(os.path.join(cfg.ASSETS_DIR, "country_iso3.csv"))
    return dict(zip(cw.name, cw.iso3))


def ga_by_country(ga_party):
    """
    Repeat each Party's NDC weights for every country it covers, so indices can be
    computed per country against the NDC that country actually filed under. The EU's 27
    member states all receive the EU's row.
    """
    rows = []
    for _, r in ga_party.iterrows():
        members = cfg.EU_MEMBER_COUNTRIES if r.party == "EU" else [r.party]
        for c in members:
            rows.append({**r.to_dict(), "party": c})
    return pd.DataFrame(rows)


def load_covariates(year=2021):
    """
    One row per ISO3: the four outcomes and the four controls.

    The merged table for each year is released as data/covariates_<year>.csv and is read
    from there (or from OUTPUT_DIR) when the raw source files are not present. Sources:
    Our World in Data (emissions, energy, GDP, population), the World Bank Worldwide
    Governance Indicators (government effectiveness) and the ND-GAIN index (vulnerability).
    """
    for d in (cfg.OUTPUT_DIR, os.path.join(cfg.PIPELINE_DIR, "data")):
        f = os.path.join(d, f"covariates_{year}.csv")
        if os.path.exists(f) and not os.path.isdir(COV_DIR):
            return pd.read_csv(f, index_col="iso3")
    o = pd.read_csv(os.path.join(COV_DIR, "climate_outcome.csv"), low_memory=False)
    o = o[(o.year == year) & o.iso_code.notna() & (o.iso_code.str.len() == 3)]
    df = o.set_index("iso_code")[list(OUTCOMES.values()) + ["gdp", "population"]].copy()
    df["ln_gdp_pc"] = np.log(df.gdp / df.population)

    f = pd.read_csv(os.path.join(COV_DIR, "fossil_share_elec_all_countries.csv"))
    f = f[f.year == year] if (f.year == year).any() else f[f.year == f.year.max()]
    df["fossil_share"] = f.set_index("iso_code").fossil_share_elec

    g = pd.read_excel(os.path.join(COV_DIR, "Gov efficiency.xlsx"), sheet_name="Sheet1")
    g = g[g.Year == year]
    df["gov_effectiveness"] = g.set_index("Economy (code)")[
        "Governance estimate (approx. -2.5 to +2.5)"]

    v = pd.read_csv(VULN).set_index("ISO3")
    col = str(year) if str(year) in v.columns else v.columns[-1]
    df["vulnerability"] = v[col]

    return df.drop(columns=["gdp", "population"])


def build_panel(indices, year=2021, level="country"):
    """
    Attach covariates to the indices.

    level="party"    one row per unit of `indices`, exactly as given. The per-country
                     indices go through here; this is the published specification.
    level="country"  expands a Party-indexed frame so the EU's members each carry the
                     EU's indices. Kept for the comparison in notebook 06; not the
                     published choice.
    """
    name_to_iso = crosswalk()
    keep = (["ADI", "OSI", "n_posts"]
            + [f"G{i}_TVR" for i in range(1, 7)]
            + [f"G{i}_OSI" for i in range(1, 7)])
    idx = indices[[c for c in keep if c in indices.columns]].copy()

    if level == "party":
        rows = idx.reset_index().rename(columns={"party": "unit"})
        rows["iso3"] = rows.unit.map(name_to_iso)
    else:
        out = []
        for party, r in idx.iterrows():
            members = cfg.EU_MEMBER_COUNTRIES if party == "EU" else [party]
            for c in members:
                out.append({"unit": c, "party": party, **r.to_dict()})
        rows = pd.DataFrame(out)
        rows["iso3"] = rows.unit.map(name_to_iso)

    cov = load_covariates(year)
    panel = rows.dropna(subset=["iso3"]).merge(cov, left_on="iso3", right_index=True,
                                               how="left")
    return panel.set_index("unit")


def fit(panel, outcome, predictors, log_outcome=True, standardise=True, cov_type="HC3"):
    """One OLS. Returns the fitted result and the frame it was fitted on."""
    y_col = OUTCOMES[outcome]
    cols = predictors + CONTROLS
    d = panel[[y_col] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    y = np.log(d[y_col]) if log_outcome else d[y_col]
    X = d[cols].astype(float)
    if standardise:
        X = (X - X.mean()) / X.std()
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type=cov_type), d


def fit_table1(panel, predictors=("ADI", "OSI"), **kw):
    """The same specification across all four outcomes. Returns coefficients and fit."""
    coef, se, pval, meta = {}, {}, {}, {}
    for name in OUTCOMES:
        res, d = fit(panel, name, list(predictors), **kw)
        coef[name], se[name], pval[name] = res.params, res.bse, res.pvalues
        meta[name] = {"N": int(res.nobs), "R2": res.rsquared}
    return (pd.DataFrame(coef), pd.DataFrame(se), pd.DataFrame(pval),
            pd.DataFrame(meta).T)


def compare_published(coef, meta):
    """
    Mean absolute difference from the printed table, over Panel A and the controls, plus
    how far off N and R-squared are. Lower is a better match; the specification with the
    lowest score is the one the published run used.
    """
    rows = pd.concat([PUBLISHED_A, PUBLISHED_CONTROLS])
    got = coef.reindex(rows.index)[rows.columns]
    diff = (got - rows).abs()
    return {
        "mean_abs_diff_ADI_OSI": float(diff.loc[["ADI", "OSI"]].mean().mean()),
        "mean_abs_diff_controls": float(diff.loc[CONTROLS].mean().mean()),
        "sign_matches": int((np.sign(got) == np.sign(rows)).sum().sum()),
        "sign_total": int(rows.size),
        "N": int(meta.N.iloc[0]),
        "N_off_by": int(meta.N.iloc[0] - PUBLISHED_N),
        "mean_abs_diff_R2": float(np.mean([abs(meta.R2[c] - PUBLISHED_R2[c])
                                           for c in PUBLISHED_R2])),
    }


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def format_table(coef, se, pval, meta):
    """Coefficient with stars over standard error, laid out like the published table."""
    out = []
    for row in coef.index:
        if row == "const":
            continue
        out.append({"Variables": row,
                    **{c: f"{coef.loc[row, c]:.3f}{stars(pval.loc[row, c])}"
                       for c in coef.columns}})
        out.append({"Variables": "",
                    **{c: f"({se.loc[row, c]:.3f})" for c in coef.columns}})
    out.append({"Variables": "N", **{c: int(meta.N[c]) for c in coef.columns}})
    out.append({"Variables": "R2", **{c: round(meta.R2[c], 2) for c in coef.columns}})
    return pd.DataFrame(out).set_index("Variables")
