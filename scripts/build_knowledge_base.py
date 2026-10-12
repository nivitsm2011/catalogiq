"""Phase 4 step 1: style guide, brand lexicon and the Chroma knowledge base."""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from catalogiq.config import get_config, get_rag_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.rag.knowledge_base import build_knowledge_base, load_listings
from catalogiq.rag.style_guide import write_style_guide

logger = get_logger("build_knowledge_base")


def build_brand_lexicon() -> list[str]:
    """Brands seen in catalog product names (leading 1-2 words) and Amazon ``store`` values."""
    cfg = get_rag_config()["guardrails"]
    ignore = set(cfg["brand_ignore"])
    train = pd.read_parquet(resolve_path("data/processed/fashion_train.parquet"))
    counts: Counter[str] = Counter()
    for name in train["productDisplayName"].dropna():
        words = str(name).split()
        for n in (1, 2, 3):
            if len(words) >= n:
                counts[" ".join(words[:n]).lower()] += 1
    brands = {b for b, c in counts.items() if c >= cfg["brand_lexicon_min_count"]}
    # keep the longest group when a longer one is as frequent (e.g. "united colors of benetton")
    brands = {b for b in brands if not (set(b.split()) & ignore)}
    brands = {
        b
        for b in brands
        if not any(
            o != b and o.startswith(b + " ") and counts[o] >= 0.8 * counts[b] for o in brands
        )
    }
    items = load_listings()
    brands |= {str(s).lower().strip() for s in items["store"].dropna() if 3 <= len(str(s)) <= 30}
    brands = {
        b
        for b in brands
        if len(b) >= 4 and not b.isdigit() and "'" not in b and not (set(b.split()) & ignore)
    }
    return sorted(brands)


def main() -> None:
    import sys

    write_style_guide()
    lex = build_brand_lexicon()
    path = resolve_path(get_rag_config()["paths"]["brand_lexicon"])
    path.write_text(json.dumps(lex, indent=0), encoding="utf-8")
    logger.info("Brand lexicon: %d entries -> %s", len(lex), path)
    if "--lexicon-only" in sys.argv:
        return
    counts = build_knowledge_base()
    logger.info("Done: %s", counts)
    _ = get_config


if __name__ == "__main__":
    main()
