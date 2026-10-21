"""
Shared configuration for the communicative-distance pipeline.

Every notebook starts with `import config as cfg`, so all paths, model names,
random seeds and fixed vocabularies live in one place. Paths default to folders
inside this repository and can be overridden with environment variables.

Package versions used for the results reported in the paper:
    python 3.11        numpy 1.26.4        pandas 2.x
    bertopic 0.16.4    sentence_transformers 3.4.1
    nltk 3.9.1         gensim 4.x          umap-learn / hdbscan
    bert_score / evaluate (BERTScore)      transformers 4.x
"""

import json
import os

# ── 1. Paths ────────────────────────────────────────────────────────────────
# The repository root (this file lives in src/).
PIPELINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(PIPELINE_DIR, "assets")

# Derived outputs. The released derived data live in data/; copy them here (or point
# CD_OUTPUT_DIR at data/) to run notebooks 05 onwards.
OUTPUT_DIR = os.environ.get("CD_OUTPUT_DIR", os.path.join(PIPELINE_DIR, "output"))

# Raw inputs, not distributed with the repository (see README).
RAW_DIR = os.environ.get("CD_RAW_DIR", os.path.join(PIPELINE_DIR, "data", "raw"))
NDC_TXT_DIR = os.environ.get("CD_NDC_DIR", os.path.join(RAW_DIR, "ndc_translated"))   # one .txt per Party
RAW_TWEET_DIR = os.environ.get("CD_TWEET_DIR", os.path.join(RAW_DIR, "x_archive"))   # Global-EC-*.json.gz
WORK_DIR = os.environ.get("CD_WORK_DIR", os.path.join(PIPELINE_DIR, "work"))

# Two legacy locations used only by optional comparison steps (the 2025 per-post
# results in WORK_DIR, raw covariate files and an older basemap under PROJECT_ROOT).
# None of these inputs is distributed; the merged covariates are in
# data/regression_panel_2022.csv.
REVISION_DIR = os.environ.get("CD_REVISION_DIR", PIPELINE_DIR)
PROJECT_ROOT = os.environ.get("CD_PROJECT_ROOT", RAW_DIR)

# ── 2. Analysis window ──────────────────────────────────────────────────────
# The vendor corpus runs 2021-12-15 to 2023-03-13, but the analysis is the
# 2022 calendar year, which is what the manuscript describes.
ANALYSIS_YEAR = 2022

# Countries (after collapsing the EU into one Party) need more than this many
# posts to enter the country-level comparison.
MIN_TWEETS_PER_PARTY = 100

# ── 3. Models ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "climatebert/distilroberta-base-climate-f"   # NDC chunks
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"              # tweet <-> topic
STANCE_MODEL = "robookwus/climate-change-belief"               # tweet stance

# BERTScore settings, kept identical to the original run
BERTSCORE_KWARGS = dict(idf=False, rescale_with_baseline=True, lang="en")

# Stance labels -> {-1, 0, +1}. The classifier emits four labels; "factual news
# on climate change" is treated as neutral, the same as "neutral".
STANCE_SCORE = {
    "believer in human-made climate change": 1,
    "denier of human-made climate change": -1,
    "neutral": 0,
    "factual news on climate change": 0,
}

SEED = 8521

# ── 4. Topic model hyperparameters ──────────────────────────────────────────
CHUNK_WORDS = 500          # NDC documents are split into 500-word chunks
EMBED_SUBCHUNK_WORDS = 20  # each chunk is embedded in 20-word pieces, then max-pooled
N_TOPICS = 39              # BERTopic nr_topics; yields 38 topics plus the -1 outlier
MIN_TOPIC_SIZE = 5
TOP_N_WORDS = 100
BIGRAM_MIN_COUNT = 200
BIGRAM_THRESHOLD = 2
MIN_WORD_FREQ = 2          # drop words occurring fewer than twice in the corpus

UMAP_KWARGS = dict(n_neighbors=15, n_components=10, metric="cosine", random_state=SEED)
HDBSCAN_KWARGS = dict(min_cluster_size=5, metric="euclidean",
                      cluster_selection_method="eom", prediction_data=True)


# ── 5. Fixed vocabularies ───────────────────────────────────────────────────
def _load(name):
    with open(os.path.join(ASSETS_DIR, name), encoding="utf-8") as fh:
        return json.load(fh)


# The 38 topic sentences. These are hand-written one-line summaries of the
# BERTopic topics and are the anchor of the whole analysis: T1-T38 and G1-G6
# in the manuscript refer to these, and BERTScore matches each post against them.
TOPIC_SENTENCES = _load("topic_sentences.json")

# G1-G6 groupings. The NDC side indexes Subtopic_0..37, the Twitter side
# indexes Topic_1..38, hence two separate dictionaries.
_groups = _load("topic_groups.json")
TOPIC_GROUPS_NDC = _groups["TOPIC_GROUPS_NDC"]
TOPIC_GROUPS_TW = _groups["TOPIC_GROUPS_TW"]
MAP_TW_TOPIC_TO_NDC_SUBTOPIC = _groups["MAP_TW_TOPIC_TO_NDC_SUBTOPIC"]

GROUP_NAMES = {
    "G1": "Emission accounting and targets",
    "G2": "Sector mitigation",
    "G3": "Adaptation",
    "G4": "Governance and regulation",
    "G5": "Social inclusion",
    "G6": "Finance and carbon markets",
}

# Country name harmonisation. The place names in the feed appear in many
# languages ("Nederland", "Украина", "تونس"), so they are mapped to English and
# then to the shapefile spelling before the EU members are collapsed to "EU".
_maps = _load("country_mappings.json")
MAPPING_NDC_TO_SHP = _maps["MAPPING_NDC_TO_SHP"]
MAPPING_TW_TO_EN = _maps["MAPPING_TW_TO_EN"]
MAPPING_EN_TO_SHP = _maps["MAPPING_EN_TO_SHP"]
EU_MEMBER_COUNTRIES = _maps["EU_MEMBER_COUNTRIES"]

# Country names are stripped out before topic modelling so that topics describe
# policy content rather than who wrote the document.
COUNTRY_STOPWORDS = {
    "afghanistan", "albania", "algeria", "andorra", "angola", "antigua and barbuda", "argentina", "armenia", "australia", "austria",
    "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados", "belarus", "belgium", "belize", "benin", "bhutan",
    "bolivia", "bosnia and herzegovina", "botswana", "brazil", "brunei darussalam", "bulgaria", "burkina faso", "burundi", "cabo verde", "cambodia",
    "cameroon", "canada", "central african republic", "chad", "chile", "china", "colombia", "comoros", "congo", "costa rica",
    "croatia", "cuba", "cyprus", "czech republic", "democratic people's republic of korea", "democratic republic of the congo", "denmark", "djibouti", "dominica", "dominican republic",
    "ecuador", "egypt", "el salvador", "equatorial guinea", "eritrea", "estonia", "eswatini", "ethiopia", "fiji", "finland",
    "france", "gabon", "gambia", "georgia", "germany", "ghana", "greece", "grenada", "guatemala", "guinea",
    "guinea-bissau", "guyana", "haiti", "honduras", "hungary", "iceland", "india", "indonesia", "iran", "iraq",
    "ireland", "israel", "italy", "jamaica", "japan", "jordan", "kazakhstan", "kenya", "kiribati", "kuwait",
    "kyrgyzstan", "lao people's democratic republic", "latvia", "lebanon", "lesotho", "liberia", "libya", "liechtenstein", "lithuania", "luxembourg",
    "madagascar", "malawi", "malaysia", "maldives", "mali", "malta", "marshall islands", "mauritania", "mauritius", "mexico",
    "micronesia", "monaco", "mongolia", "montenegro", "morocco", "mozambique", "myanmar", "namibia", "nauru", "nepal",
    "netherlands", "new zealand", "nicaragua", "niger", "nigeria", "north macedonia", "norway", "oman", "pakistan", "palau",
    "panama", "papua new guinea", "paraguay", "peru", "philippines", "poland", "portugal", "qatar", "republic of korea", "republic of moldova",
    "romania", "russian federation", "rwanda", "saint kitts and nevis", "saint lucia", "saint vincent and the grenadines", "samoa", "san marino", "sao tome and principe", "saudi arabia",
    "senegal", "serbia", "seychelles", "sierra leone", "singapore", "slovakia", "slovenia", "solomon islands", "somalia", "south africa",
    "south sudan", "spain", "sri lanka", "sudan", "suriname", "sweden", "switzerland", "syrian arab republic", "tajikistan", "thailand",
    "timor-leste", "togo", "tonga", "trinidad and tobago", "tunisia", "turkey", "turkmenistan", "tuvalu", "uganda", "ukraine",
    "united arab emirates", "united kingdom of great britain and northern ireland", "united republic of tanzania", "united states of america", "uruguay", "uzbekistan", "vanuatu", "venezuela", "viet nam", "yemen", "zambia", "zimbabwe",
    "paris", "brasília", "washington",
}

# Boilerplate that survives cleaning and would otherwise dominate the topics:
# units, URL fragments, document furniture, and country abbreviations.
DOMAIN_STOPWORDS = [
    'mw', 'southafrica', 'ha', 'hectare', 'area', 'eq', 'ggco', 'total', 'nggp', 'xcd', 'viet', 'nam',
    'de', 'httpwww', 'pdf', 'org', 'bolivarianrepublic', 'caracas', 'http', 'gazette', 'table', 'py', 'eu',
    'regulationeu', 'conclusion', 'european', 'articleparagraph', 'contributestowards', 'partyaddressed',
    'find', 'explained', 'measured', 'europeanunion', 'eqkt', 'ktco', 'coeq', 'ictu', 'yet', 'ti', 'dprk', 'fondoin',
    'elsalvador', 'km', 'required', 'mg', 'kg', 'yr', 'borne', 'uk', 'gov', 'scotland', 'scottish', 'axis', 'mppee', 'tr',
    'euro', 'sendai', 'mtcoe', 'uae', 'round', 'highhigh', 'mediummedium', 'high', 'medium', 'low', 'eg', 'pd', 'solomon',
    'honiara', 'yuan', 'trim', 'iii', 'figure', 'png', 'sheet', 'jpg', 'download', 'mitigationco', 'antiguabarbuda',
    'skopje', 'trp', 'retrieved', 'nacional', 'mo', 'en', 'da', 'dela', 'maputo', 'gob', 'additional',
    'referenceindicator', 'dominicanrepublic', 'drr link', 'obtenido', 'la', 'clim', 'del', 'www', 'cambio', 'december',
    'doi', 'accessed', 'report', 'tico', 'file',
]


def stopwords_set():
    """
    Build the full stop-word set used for both NDC documents and posts:
    NLTK English stop words, plus country and capital-city names.
    """
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    try:
        words = nltk_stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        words = nltk_stopwords.words("english")
    return set(words).union(COUNTRY_STOPWORDS)


def tw_topic_to_bertopic_index(t):
    """
    T1-T38 in the manuscript are a manual renumbering of BERTopic's own 0..37
    topic indices, done when the topics were read and grouped by hand. The
    permutation is stored in MAP_TW_TOPIC_TO_NDC_SUBTOPIC; this returns the
    BERTopic index that topic T{t} corresponds to.

    Sanity check: applying it to TOPIC_GROUPS_TW reproduces TOPIC_GROUPS_NDC
    exactly, for all six groups.
    """
    return int(MAP_TW_TOPIC_TO_NDC_SUBTOPIC[f"Topic_{t}"].split("_")[1])


def ndc_columns():
    """Column names for the 38 NDC topic probabilities."""
    return [f"Subtopic_{i}" for i in range(38)]


def tw_columns():
    """Column names for the 38 tweet-topic recall scores."""
    return [f"Topic_{i}" for i in range(1, 39)]
