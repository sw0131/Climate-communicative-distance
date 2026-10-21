"""Helpers for the descriptive tables the referees asked for.

The heavy part is counting how often each Global Pulse phrase appears, which Referee 2
asks for (R2.1) and which nobody has reported for this corpus. The phrase lists are
read straight out of `climate_filter.py` rather than retyped, so the table can never
drift from the filter that actually ran.
"""

import ast
import os
import re
from collections import Counter

import numpy as np
import pandas as pd

import climate_filter

# Rules that name climate outright, as opposed to the conjunctions that need a
# climate word plus a topical word. The split matters because hand labelling put
# precision at 82.8% for the first group and 50.0% for the second.
STRONG_RULES = ["matches_general", "matches_arctic", "matches_hashtag_current",
                "matches_hashtag_campaigns", "matches_hashtag_negotiations",
                "matches_hashtag_related"]


def taxonomy_phrases():
    """
    Pull the phrase lists out of climate_filter.py, tagged with the rule they belong
    to and whether they are a positive term, a required co-occurrence, or a veto.

    Reading the source rather than restating the lists means this cannot fall out of
    step with the filter.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(climate_filter.__file__)),
                            "climate_filter.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    rows = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if not fn.name.startswith("matches_"):
            continue
        for node in ast.walk(fn):
            if not (isinstance(node, ast.Assign)
                    and isinstance(node.targets[0], ast.Name)):
                continue
            name = node.targets[0].id
            try:
                phrases = ast.literal_eval(node.value)
            except Exception:
                continue
            if not isinstance(phrases, list):
                continue
            role = ("veto" if "neg" in name or name.startswith("not_have")
                    else "required with a climate word" if name.startswith("must_have_2")
                    else "positive")
            for p in phrases:
                rows.append({"rule": fn.name.replace("matches_", ""),
                             "list": name, "role": role, "phrase": p,
                             "kind": "strong" if fn.name in STRONG_RULES else "conjunction"})
    return pd.DataFrame(rows)


def count_phrases(texts, phrases, chunk=50000):
    """
    How many posts contain each phrase. Case-insensitive substring match, the same
    test the filter makes, so the counts describe the filter's own vocabulary rather
    than an approximation of it.
    """
    counts = Counter()
    phrases = list(dict.fromkeys(phrases))
    for start in range(0, len(texts), chunk):
        block = [t.lower() for t in texts[start:start + chunk]]
        for p in phrases:
            pl = p.lower()
            counts[p] += sum(1 for t in block if pl in t)
    return pd.Series(counts, name="posts")


# Words that mark an account as something other than a private individual. Used only
# to order the top-accounts table for hand checking, never as a final label: the
# categories in the table are set by eye, and the automated guess is there to make
# that faster.
ORG_HINTS = {
    "news": ["news", "journalist", "reporter", "editor", "broadcast", "radio", "tv ",
             "magazine", "press", "media", "headlines", "bulletin"],
    "jobs": ["job", "hiring", "career", "recruit", "vacanc", "employment", "staffing",
             "apply now", "we're hiring"],
    "government": ["ministry", "minister", "government", "official account", "embassy",
                   "council", "municipal", "department of", "agency", "parliament",
                   "senator", "congress", "mp for", "mayor"],
    "ngo_research": ["ngo", "non-profit", "nonprofit", "charity", "institute",
                     "university", "research", "foundation", "society", "association",
                     "centre for", "center for", "think tank", "professor", "phd"],
    "business": ["ltd", "inc.", "llc", "company", "we provide", "our services",
                 "solutions", "consultancy", "consulting", "official page", "shop",
                 "store", "hotel", "restaurant"],
}

AUTOMATION_SOURCES = {
    "CareerArc 2.0", "dlvr.it", "IFTTT", "Hootsuite Inc.", "TweetDeck", "Buffer",
    "Zapier.com", "SocialFlow", "Paper.li", "twittbot.net", "Tide Bot", "Google",
    "WordPress.com", "Sprout Social", "Later", "Zoho Social", "TrendsMap",
}


def guess_account_type(description, screen_name="", source=""):
    """A first guess at what kind of account this is, to be checked by hand."""
    if source in AUTOMATION_SOURCES:
        return "automated"
    text = f"{description} {screen_name}".lower()
    for label, hints in ORG_HINTS.items():
        if any(h in text for h in hints):
            return label
    return "individual?"


def posting_rate(statuses_count, user_created_at, as_of="2022-07-01"):
    """
    Posts per day since the account opened. A standard automation signal, and one of
    the things Referee 1 asks about (R1.3.c).
    """
    created = pd.to_datetime(user_created_at, format="%a %b %d %H:%M:%S %z %Y",
                             errors="coerce", utc=True)
    days = (pd.Timestamp(as_of, tz="UTC") - created).dt.total_seconds() / 86400
    return pd.to_numeric(statuses_count, errors="coerce") / days.where(days > 0)
