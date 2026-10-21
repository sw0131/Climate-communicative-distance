"""The Global Pulse climate filter, lifted verbatim from
Code/Climate_twitter_identification.ipynb (cells 3-5).

Kept byte-identical to the code that produced the 399,891-tweet sample, so that
running it on `extended_tweet.full_text` measures exactly what truncation cost us.
Do not "improve" it.
"""

def any_of_phrases_in_text(text, phrases):
    """
    Returns True if 'text' contains at least one of the phrases in 'phrases'.
    Matching is case-insensitive and looks for the phrase as a substring.
    """
    for phrase in phrases:
        if phrase.lower() in text:
            return True
    return False

def none_of_phrases_in_text(text, phrases):
    """
    Returns True if NONE of the phrases in 'phrases' appear in 'text'.
    """
    return not any_of_phrases_in_text(text, phrases)


def matches_general(text):
    # (interaction.content any "climate change, global warming, ..." ) OR
    # (interaction.content any "drought, droughts" AND NOT any "goal, goals, ...")
    sub_a_phrases = [
        "climate change", "global warming", "climatology", "extreme climate", "extreme weather",
        "rising temperatures", "rising temperature", "hot planet", "hotter planet", "warming planet",
        "heating planet", "ozone layer", "price of carbon", "price of emissions", "charge for emissions",
        "cost of emissions", "sea level", "sea-level", "ocean level", "greenhouse gas", "green house gas",
        "greenhouse gases", "green house gases", "fossil fuel", "fossil fuels", "cdm procedures",
        "sustainable forest management", "cancun decision", "copenhagen accord", "kyoto protocol",
        "kyoto agreement", "energy tax", "transport tax", "aviation tax", "energy taxes",
        "transport taxes", "aviation taxes", "thermohaline", "hydrofluorocarbons", "bonn agreements"
    ]
    sub_b_phrases_pos = ["drought", "droughts"]
    sub_b_phrases_neg = [
        "goal", "goals", "scoring", "title", "titles", "sex", "rooney", "nigga",
        "niggas", "championship", "goalless", "win"
    ]

    condition_a = any_of_phrases_in_text(text, sub_a_phrases)
    condition_b = any_of_phrases_in_text(text, sub_b_phrases_pos) and none_of_phrases_in_text(text, sub_b_phrases_neg)

    return condition_a or condition_b

def matches_risk_disaster(text):
    # (interaction.content any "climate, enviro, environment") AND
    # (interaction.content any "flood, flooding, floods, ..., sink")
    must_have_1 = ["climate", "enviro", "environment"]
    must_have_2 = [
        "flood", "flooding", "floods", "flooded", "hunger", "poverty", "disaster", "disasters",
        "earthquake", "dying", "death", "deaths", "collapse", "tsunami", "tsunamis", "kill",
        "killing", "killer", "killers", "tornado", "tornados", "pollution", "polluting",
        "polluters", "pollutant", "pollutants", "famine", "smog", "disease", "diseases",
        "wildfires", "fire", "fires", "smoke", "windfall", "survive", "destroying", "destroys",
        "destroyed", "hurricane", "hurricanes", "sealife", "acidification", "disruption",
        "dilemma", "trilemma", "solving", "solution", "uninhabitable", "endanger", "endangered",
        "endangering", "catastrophic", "catastrophe", "catastrophes", "fail", "failed", "failing",
        "failure", "triggers", "triggered", "set off", "trigger", "changes", "changed",
        "changing", "lost", "losing", "losers", "lose", "irreversible", "model", "models",
        "modeling", "modelling", "modelled", "modeled", "mitigate", "mitigation", "mitigates",
        "adaptation", "adapt", "adapts", "adaptations", "rise", "risen", "rises", "rising",
        "shifting", "shifted", "shifts", "damage", "damages", "damaged", "damaging", "toxic",
        "toxicity", "crisis", "crises", "health", "intensity", "intensify", "intensifies",
        "sink", "sinks", "sinking"
    ]

    return any_of_phrases_in_text(text, must_have_1) and any_of_phrases_in_text(text, must_have_2)

def matches_politics_opinion(text):
    # (interaction.content any "climate, enviro, environment, carbon") AND NOT (monoxide, fork, etc.)
    # AND (big list of political terms)
    must_have_1 = ["climate", "enviro", "environment", "carbon"]
    not_have_1 = ["monoxide", "fork", "copy", "fiber"]
    must_have_2 = [
        "rio+20", "rio +20", "rio convention", "g8", "group of eight", "negotiation", "activism",
        "activists", "politics", "political", "government", "global", "international", "conference",
        "stakeholder", "accord", "ratify", "gender", "unep", "undp", "united nations", "mdg",
        "development", "rant", "region", "human rights", "science", "scientist", "skeptic",
        "denial", "conspiracy", "debate", "econ", "economy", "manmade", "commitment", "target",
        "warning", "humanity", "greenland", "proof", "myth", "hoax", "scam", "security",
        "technology", "pro-science", "research", "opportunity", "youth", "dialogue", "progress",
        "sustainable", "volatility", "cooperation", "protest", "leaders", "reduce", "waste",
        "planet", "conservation", "peace", "timetable", "obama", "president", "congress",
        "prime minister", "pledge", "brazil", "china", "india", "russia", "bric"
    ]

    if not any_of_phrases_in_text(text, must_have_1):
        return False
    if any_of_phrases_in_text(text, not_have_1):
        return False
    return any_of_phrases_in_text(text, must_have_2)

def matches_economy(text):
    # (interaction.content any "climate, enviro, environment, carbon") AND NOT ...
    # AND any of the big economy terms
    must_have_1 = ["climate", "enviro", "environment", "carbon"]
    not_have_1 = ["monoxide", "fork", "copy", "fiber"]
    must_have_2 = [
        "price", "pricing", "tax", "market", "finance", "financial", "invest", "investment",
        "fund", "budget", "stock", "economist", "business", "industry", "industrial", "consumption",
        "cap-and-trade", "footprint", "sequestration", "trade", "transport", "technology",
        "abatement", "assessment", "measurement", "baseline", "capture"
    ]
    if not any_of_phrases_in_text(text, must_have_1):
        return False
    if any_of_phrases_in_text(text, not_have_1):
        return False
    return any_of_phrases_in_text(text, must_have_2)

def matches_energy(text):
    # (interaction.content any "climate, enviro, environment, carbon") AND NOT ...
    # AND any of [fuel, fueling, solar, ...]
    # OR (climate/enviro/environment) AND 'contains "carbon"'
    must_have_1 = ["climate", "enviro", "environment", "carbon"]
    not_have_1 = ["monoxide", "fork", "copy", "fiber"]

    sub_a_phrases = [
        "fuel", "fueling", "fuelled", "fuels", "offshore", "solar", "renewable", "extraction",
        "energy", "drilling", "oil", "oilfields", "gas", "emission", "emissions", "wind power",
        "solar power", "pv power", "pv cells", "photovoltaic", "coal", "mine", "mining", "turbine",
        "fracking", "methane", "biofuel", "co2"
    ]
    # Condition 1
    cond1 = (
        any_of_phrases_in_text(text, must_have_1) and
        not any_of_phrases_in_text(text, not_have_1) and
        any_of_phrases_in_text(text, sub_a_phrases)
    )
    # Condition 2
    cond2_must_have = ["climate", "enviro", "environment"]
    cond2 = (
        any_of_phrases_in_text(text, cond2_must_have) and
        not any_of_phrases_in_text(text, not_have_1) and
        ("carbon" in text.lower())
    )
    return cond1 or cond2

def matches_oceans_water(text):
    # (interaction.content any "climate, enviro, environment") AND
    # (interaction.content any "underwater, seas, sea, aquifer, water, iceberg, ocean")
    must_have_1 = ["climate", "enviro", "environment"]
    must_have_2 = [
        "underwater", "under water", "seas", "sea", "aquifer", "aquifers", "rivers", "river",
        "water", "iceberg", "icebergs", "ocean", "oceans"
    ]
    return any_of_phrases_in_text(text, must_have_1) and any_of_phrases_in_text(text, must_have_2)

def matches_weather(text):
    # (interaction.content any "climate, enviro, environment") AND
    # (interaction.content any "weather, heat, desert, hot, atmosphere, melt, etc.")
    must_have_1 = ["climate", "enviro", "environment"]
    must_have_2 = [
        "weather", "heat", "heatwave", "heatstroke", "desert", "desertification", "hot",
        "heating", "heated", "hotter", "hottest", "atmosphere", "atmospheric", "boil",
        "coastal", "coast", "warmer", "warming", "warmest", "melt", "melted", "melting",
        "ozone", "ice", "icy", "cracking", "arctic", "icecap", "icecaps", "glaciers",
        "north pole", "south pole", "polar", "penguin", "penguins", "volatility", "fluctuate",
        "cycle", "temperature", "temps", "temp", "rain", "snow", "precipitation", "ghg",
        "record", "recorded", "summer", "winter", "spring", "fall", "autumn"
    ]
    return any_of_phrases_in_text(text, must_have_1) and any_of_phrases_in_text(text, must_have_2)

def matches_agriculture_forestry(text):
    # (interaction.content any "climate, enviro, environment, carbon") AND NOT ...
    # AND any of [hectare, acres, forest, reforestation, etc.]
    must_have_1 = ["climate", "enviro", "environment", "carbon"]
    not_have_1 = ["monoxide", "fork", "copy", "fiber"]
    must_have_2 = [
        "hectare", "hectares", "acre", "acres", "acreage", "forest", "forests", "forestation",
        "forestry", "deforestation", "reforestation", "ecosystem", "ecosystems", "ecology",
        "ecologies", "eco", "tree", "trees", "farm", "farmer", "farming", "flora", "plants",
        "plant", "planted", "planting", "grass", "green", "greening", "biomass", "biome",
        "biodiversity", "bio", "biological", "biologist", "land use", "land right", "land rights",
        "crop", "crops", "monsanto"
    ]
    if not any_of_phrases_in_text(text, must_have_1):
        return False
    if any_of_phrases_in_text(text, not_have_1):
        return False
    return any_of_phrases_in_text(text, must_have_2)

def matches_arctic(text):
    # (interaction.content any "Arctic, North Pole, ice caps, glacier, polar ice")
    # AND (interaction.content any "melt, melted, melting, disappear, vanish, gone")
    must_have_1 = [
        "arctic", "north pole", "south pole", "ice caps", "ice cap", "global ice",
        "glacial", "glacier", "glaciers", "polar ice"
    ]
    must_have_2 = [
        "melt", "melted", "melting", "disappear", "disappearing", "vanish", "vanishing", "gone"
    ]
    return any_of_phrases_in_text(text, must_have_1) and any_of_phrases_in_text(text, must_have_2)


###############################################################################
# HASHTAG MATCHING
###############################################################################

def matches_hashtag_current(text):
    hashtags = [
        "#climatechange", "#globalwarming", "#climate", "#climateaction", "#climateambition",
        "#space4ambition", "#m4c", "#priceoncarbon", "#climatemarch", "#peoplesclimate",
        "#peoplesclimatemarch"
    ]
    return any_of_phrases_in_text(text, hashtags)

def matches_hashtag_campaigns(text):
    hashtags = [
        "#tcktcktck", "#fossilfree", "#actonclimate", "#connect4climate", "#manystrongvoices"
    ]
    return any_of_phrases_in_text(text, hashtags)

def matches_hashtag_negotiations(text):
    hashtags = [
        "#climate2022", "#paris2015", "#adp2022", "#ipcc", "#ar6", "#ar5", "#cop27","#cop26","#cop25","#cop24","#cop23","#cop22","#cop21", "#cop20",
        "#cop19", "#cop18", "#cop17", "#cop16", "#cop15", "#cop14", "#cop13", "#cop12",
        "#cop11", "#cop10", "#cop9", "#cop8", "#cop7", "#cop6", "#cop5"
    ]
    return any_of_phrases_in_text(text, hashtags)

def matches_hashtag_related(text):
    hashtags = [
        "#climateforests", "#climateresilience", "#climatefinance", "#capacitybuilding",
        "#climatedeparture", "#climatedata", "#climatehoax", "#climatetaxfraud", "#extremeweather",
        "#climateproofmyland", "#carboncapture", "#sustainability", "#renewableenergy", "#gogreen",
        "#cleanpower", "#cleantech", "#cleanenergy", "#risingseas", "#fossilfuels", "#pollution",
        "#climatetraining"
    ]
    return any_of_phrases_in_text(text, hashtags)


def is_climate_tweet(text):
    """
    Returns True if `text` matches ANY of the climate-related topics or hashtags,
    replicating the CSDL-style OR logic.
    """
    # Check hashtags
    if matches_hashtag_current(text):
        return True
    if matches_hashtag_campaigns(text):
        return True
    if matches_hashtag_negotiations(text):
        return True
    if matches_hashtag_related(text):
        return True
    # Check key words
    if matches_general(text):
        return True
    if matches_risk_disaster(text):
        return True
    if matches_politics_opinion(text):
        return True
    if matches_economy(text):
        return True
    if matches_energy(text):
        return True
    if matches_oceans_water(text):
        return True
    if matches_weather(text):
        return True
    if matches_agriculture_forestry(text):
        return True
    if matches_arctic(text):
        return True
    # If none matched
    return False
