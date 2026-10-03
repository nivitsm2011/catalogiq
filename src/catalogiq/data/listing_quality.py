"""Listing quality score used to pick well-performing, real listings as RAG exemplars.

Score = Bayesian-shrunk average rating, rescaled to 0-1::

    shrunk = (v / (v + m)) * R + (m / (v + m)) * C
    score  = (shrunk - 1) / 4

where R is the item's average rating, v its number of ratings, C the catalog-wide mean rating and
m a prior strength (``listing_quality.prior_strength``). A 5.0 item with 3 ratings is pulled toward
the catalog mean, while a 4.6 item with 2,000 ratings barely moves, so "high rating x enough
ratings" is a single number. An item is an *exemplar* only if it also has a decent score, at least
``exemplar_min_ratings`` ratings and a complete listing (title length, feature bullets), because we
want the generator to imitate listings that both perform and are well written.
"""

from __future__ import annotations

import pandas as pd


def add_quality_score(items: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add ``quality_score``, ``is_complete`` and ``is_exemplar`` columns.

    Args:
        items: Cleaned item metadata with ``average_rating``, ``rating_number``, ``title``,
            ``features``.
        cfg: ``listing_quality`` config section.

    Returns:
        Copy of ``items`` with the new columns.
    """
    out = items.copy()
    prior_mean = out["average_rating"].mean()
    v = out["rating_number"].fillna(0)
    m = cfg["prior_strength"]
    shrunk = (v / (v + m)) * out["average_rating"] + (m / (v + m)) * prior_mean
    out["quality_score"] = ((shrunk - 1) / 4).clip(0, 1)
    out["is_complete"] = (out["title"].str.len() >= cfg["exemplar_min_title_chars"]) & (
        out["features"].map(len) >= cfg["exemplar_min_features"]
    )
    out["is_exemplar"] = (
        (out["quality_score"] >= cfg["exemplar_min_score"])
        & (v >= cfg["exemplar_min_ratings"])
        & out["is_complete"]
    )
    return out
