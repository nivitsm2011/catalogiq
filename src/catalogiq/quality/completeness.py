"""Listing completeness: how many of the attributes the category requires are filled in.

The required attributes per category come from the same config as the Phase 4 style guide
(``configs/rag.yaml``), plus the fields in ``quality.completeness.always_required``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from catalogiq.config import get_quality_config
from catalogiq.rag.categories import group_from_article_type, group_spec

EMPTY = {"", "nan", "none", "unknown", "n/a"}


def is_missing(value: Any) -> bool:
    """True for None, NaN, empty text and placeholder words."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return True
    return str(value).strip().lower() in EMPTY


def required_attributes(article_type: str | None) -> tuple[str, list[str]]:
    """(category group, required attribute names) for a product type."""
    group = group_from_article_type(article_type or "")
    required = list(group_spec(group)["required_attributes"])
    for extra in get_quality_config()["completeness"]["always_required"]:
        if extra not in required:
            required.append(extra)
    return group, required


def completeness(listing: dict[str, Any]) -> dict[str, Any]:
    """Completeness of one listing.

    Returns:
        ``group``, ``required``, ``missing`` and ``pct`` (0-100, share of required attributes
        present).
        A missing product type is itself reported as missing and uses the default category.
    """
    group, required = required_attributes(listing.get("articleType"))
    missing = [a for a in required if is_missing(listing.get(a))]
    pct = 100.0 * (len(required) - len(missing)) / len(required)
    return {"group": group, "required": required, "missing": missing, "pct": pct}
