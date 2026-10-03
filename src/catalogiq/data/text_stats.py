"""Simple keyword statistics on listings and reviews (explainable, no black boxes)."""

from __future__ import annotations

import re

import pandas as pd

# Attribute families a good fashion listing mentions. Plain keyword lists keep this transparent.
ATTRIBUTE_KEYWORDS: dict[str, list[str]] = {
    "material": [
        "cotton",
        "polyester",
        "leather",
        "wool",
        "linen",
        "silk",
        "denim",
        "nylon",
        "rayon",
        "spandex",
        "fleece",
    ],
    "fit": ["slim", "regular fit", "relaxed", "loose", "stretch", "true to size", "oversized"],
    "care": ["machine wash", "hand wash", "dry clean", "imported"],
    "occasion": [
        "casual",
        "formal",
        "party",
        "work",
        "office",
        "everyday",
        "sport",
        "outdoor",
        "wedding",
    ],
    "size_info": ["size", "xs", "small", "medium", "large", "xl"],
}

REVIEW_TOPICS: dict[str, list[str]] = {
    "fit": ["fit", "fits", "tight", "loose", "snug", "baggy"],
    "size": ["size", "sizing", "small", "large", "runs", "xl", "xs"],
    "fabric / material": [
        "fabric",
        "material",
        "cotton",
        "polyester",
        "leather",
        "thin",
        "soft",
        "itchy",
    ],
    "colour": ["color", "colour", "faded", "bright", "shade"],
    "quality / durability": [
        "quality",
        "cheap",
        "durable",
        "flimsy",
        "ripped",
        "tore",
        "fell apart",
        "stitch",
    ],
    "comfort": ["comfortable", "comfy", "uncomfortable", "comfort"],
    "price / value": ["price", "worth", "value", "expensive", "money"],
    "shipping / delivery": ["shipping", "delivery", "arrived", "packaging", "late"],
}


def _mentions(text: pd.Series, words: list[str]) -> pd.Series:
    """Boolean Series: does each text contain any keyword as a whole word (one regex)."""
    pattern = r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b"
    return text.str.contains(pattern, regex=True)


def listing_text(row: pd.Series) -> str:
    """Concatenate title, bullets and description into one lowercase string."""
    return " ".join([row["title"], " ".join(row["features"]), row["description"]]).lower()


def listing_profile(items: pd.DataFrame) -> pd.DataFrame:
    """Per-item copy stats: title length, bullet count, description length, attribute mentions."""
    out = pd.DataFrame(index=items.index)
    out["title_chars"] = items["title"].str.len()
    out["n_bullets"] = items["features"].map(len)
    out["description_chars"] = items["description"].str.len()
    text = items.apply(listing_text, axis=1)
    for family, words in ATTRIBUTE_KEYWORDS.items():
        out[f"mentions_{family}"] = _mentions(text, words)
    out["n_attribute_families"] = out[[c for c in out if c.startswith("mentions_")]].sum(axis=1)
    return out


def topic_share(reviews: pd.DataFrame, topics: dict[str, list[str]] | None = None) -> pd.Series:
    """Share of reviews mentioning each topic's keywords."""
    topics = topics or REVIEW_TOPICS
    text = (reviews["title"].fillna("") + " " + reviews["text"]).str.lower()
    return pd.Series({name: _mentions(text, words).mean() for name, words in topics.items()})
