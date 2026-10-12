"""Fashion category groups: keyword rules for Amazon titles, article-type map for Kaggle.

The Amazon ``categories`` field is empty for every sampled item, so a listing's group is derived
from its title with the ordered keyword rules in ``configs/rag.yaml``. The first matching rule
wins, which is why specific groups (watches, eyewear, socks) come before generic ones (topwear).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from catalogiq.config import get_rag_config


@lru_cache(maxsize=1)
def _compiled_rules() -> list[tuple[str, re.Pattern[str]]]:
    groups = get_rag_config()["categories"]["groups"]
    return [
        (name, re.compile("|".join(spec["keywords"]), re.IGNORECASE))
        for name, spec in groups.items()
        if spec["keywords"]
    ]


@lru_cache(maxsize=1)
def article_type_map() -> dict[str, str]:
    """Kaggle ``articleType`` -> category group."""
    groups = get_rag_config()["categories"]["groups"]
    return {at: name for name, spec in groups.items() for at in spec["article_types"]}


def group_from_title(title: str) -> str:
    """Category group of an Amazon listing from its title (first matching keyword rule)."""
    for name, pattern in _compiled_rules():
        if pattern.search(title or ""):
            return name
    return get_rag_config()["categories"]["default_group"]


def group_from_article_type(article_type: str) -> str:
    """Category group of a Kaggle article type (``other`` when it is not mapped)."""
    return article_type_map().get(article_type, get_rag_config()["categories"]["default_group"])


def group_spec(group: str) -> dict[str, Any]:
    """Config entry (required attributes, guidance) of one group; defaults for ``other``."""
    groups = get_rag_config()["categories"]["groups"]
    return groups.get(group, {"required_attributes": ["articleType"], "guidance": ""})
