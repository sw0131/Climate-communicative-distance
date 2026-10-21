# Communicative distance between national climate pledges and online climate discourse

Code and derived data for the manuscript *Social Media Derived Communicative Distance
Diverges from National Climate Pledges* (Communications Earth & Environment,
COMMSENV-26-3931). The manuscript is under double-anonymized review; author information
and a citation will be added on publication.

The study places Nationally Determined Contributions (NDCs) and climate-related posts
from X in a shared topic space of 38 topics (T1-T38) grouped into six thematic groups
(G1-G6), and compares them with three indices: the Attention Distance Index (ADI), the
Online Stance Index (OSI) and the Topic Visibility Ratio (TVR). Definitions are in the
Methods of the manuscript.

## Layout

| Path | Contents |
|---|---|
| `01_ndc_pipeline.ipynb` | NDC preprocessing, BERTopic topic model (frozen, see below), country-level NDC topic weights |
| `02_tweets_filter.ipynb` | climate-related posts from the raw X archive, using the full post text |
| `03_tweets_topic.ipynb` | BERTScore recall of each post against the 38 topic sentences |
| `04_tweets_stance.ipynb` | stance classification |
| `05_aggregate.ipynb` | country-level ADI, OSI and TVR |
| `06_regressions.ipynb` | OLS models of Table 2 (HC3 standard errors) |
| `07_descriptives.ipynb` | descriptive tables of Supplementary Note 1 |
| `08_sensitivity.ipynb` | sensitivity analyses of Supplementary Note 3 |
| `09_figure1_sankey.ipynb`, `10_figure2_map.ipynb`, `11_figure3_maps.ipynb`, `12_figure4_quadrants.ipynb` | Figures 1 to 4 |
| `src/` | the modules the notebooks import (`config.py` holds every path, model name, seed and fixed vocabulary); each notebook adds `src/` to `sys.path` in its first cell |
| `assets/` | fixed inputs: the 38 topic sentences, topic-to-group mapping, topic keywords, country-name mappings and ISO3 codes, figure labels, Natural Earth 1:50m basemap (public domain) |
| `data/` | derived data released with the paper (see next section) |
| `models/Bert_topic_model` | the fitted BERTopic model |

## Released data

`data/` mirrors the `output/` folder the notebooks write to.

- `indices_2022_by_country.csv`: ADI, OSI, and group- and topic-level TVR and OSI for the 100 countries in the analytical sample.
- `regression_panel_2022.csv`: the same indices with the covariates used in Table 2 (96 countries with complete covariates).
- `covariates_2021.csv` and `covariates_2022.csv`: the four outcomes and four controls by ISO3 code, merged from Our World in Data, the Worldwide Governance Indicators and ND-GAIN. Table 2 uses the 2021 values, the latest year with near-complete coverage.
- `GA_by_country.csv`: NDC topic weights by country. `topic_table.csv`: the 38 topics with their sentences, groups and keywords. `country_totals.csv`: posts and climate-related posts by country code in the 2022 archive.
- `SI/`: Table 1 of the main text, Supplementary Tables 1, 2 and 5 to 8, the sample funnel, and the source data of Figures 2 to 4. Supplementary Tables 3 and 4 quote posts and are therefore only in the Supplementary Information.
- `submission/`: `Source Data.xlsx` (one sheet per main and supplementary figure) and `Supplementary Data 1` and `2` as submitted.

## What can be reproduced from the released data

- Table 2: `06_regressions.ipynb` from `data/indices_2022_by_country.csv` and `data/covariates_2021.csv`; it writes `output/tables/Table2.csv`.
- Figures 2 and 3: `10_figure2_map.ipynb` and `11_figure3_maps.ipynb` from `data/indices_2022_by_country.csv`.
- Notebooks 02 to 05, 07, 08, 09 and 12 read post-level files (the posts, and the topic recall and stance of each post) that are not distributed; the values they produce are in `data/SI` and `data/submission/Source Data.xlsx`.

To run a notebook, copy `data/` to `output/` (or set `CD_OUTPUT_DIR` to the `data` folder) and start Jupyter from the repository root, so that the `src/` path in the first cell resolves.

Notebook outputs are kept so that the results can be inspected without the restricted post-level data: notebooks 06, 10 and 11 carry the outputs of a fresh run from the released data, and notebooks 05, 08, 09 and 12 carry the outputs of the run that produced the paper. Notebooks 01 to 04 were run on a GPU machine and have no saved outputs.

## Data not distributed

- **X posts.** The archived posts, and the per-post topic recall and stance derived from them, cannot be redistributed under the platform's data-use terms.
- **NDC texts.** The documents are public (UNFCCC NDC Registry, Climate Watch). To rerun notebook 01 from scratch, place the translated texts, one `.txt` per Party, in `data/raw/ndc_translated/` or set `CD_NDC_DIR`.
- **Covariates.** Our World in Data, the Worldwide Governance Indicators and the ND-GAIN index are public; the merged country-year values are in `data/covariates_2021.csv` and `data/covariates_2022.csv`.

## The topic model is frozen

Notebook 01 ships with `RERUN_TOPIC_MODEL = False`. It keeps the code that produced the
topics but loads the fitted model from `models/Bert_topic_model` instead of refitting.
UMAP and HDBSCAN depend on the numerical libraries underneath them, so the same seed on a
different machine can give a different clustering and therefore a different topic
numbering, on which T1-T38, G1-G6, Table 1 and Figure 1 all depend. Everything after
notebook 01 takes the topics as given.

T1-T38 are a manual renumbering of BERTopic's own indices. The permutation is in
`config.py` (`MAP_TW_TOPIC_TO_NDC_SUBTOPIC`); use `cfg.tw_topic_to_bertopic_index()` before
pulling anything out of the fitted model, or keywords attach to the wrong topic.

## Environment

Notebooks 01 (as shipped), 05-11 and the figure scripts need only the packages in
`requirements.txt`. The post-side steps (notebooks 02-04) and refitting the topic model
need a GPU and the pinned stack used for the paper:

```bash
conda create -n climate python=3.11 -y
conda activate climate
pip install numpy==1.26.4 pandas nltk==3.9.1 gensim==4.4.0 \
            bertopic==0.16.4 sentence_transformers==3.4.1 \
            umap-learn hdbscan torch transformers bert_score evaluate
```

Models: `climatebert/distilroberta-base-climate-f` (NDC embeddings),
`microsoft/deberta-xlarge-mnli` (BERTScore; `src/download_model.sh` fetches it),
`robookwus/climate-change-belief` (stance). Regressions use statsmodels 0.14.5.
