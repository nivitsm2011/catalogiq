"""Streaming, sampling and cleaning of Amazon Reviews 2023 (Amazon_Fashion).

The full dump is never stored: JSONL files are streamed from the Hugging Face Hub and only the
selected items' records are kept in memory and then written as small parquet files.
"""

from __future__ import annotations

import html
import json
import random
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def stream_records(url: str) -> Iterator[dict[str, Any]]:
    """Stream a remote JSONL file record by record with ``datasets`` streaming mode.

    The file is read as raw text lines and parsed with ``json`` (one line = one record) so that
    Amazon's irregular nested fields (``details``, mixed-type ``price``) cannot break Arrow
    schema inference. Nothing is written to disk.
    """
    ds = load_dataset("text", data_files=url, split="train", streaming=True)
    for row in ds:
        line = row["text"]
        if line:
            yield json.loads(line)


def strip_html(text: str | None) -> str:
    """Remove HTML tags/entities and collapse whitespace."""
    if not text:
        return ""
    text = _TAG_RE.sub(" ", html.unescape(str(text)))
    return _WS_RE.sub(" ", text).strip()


def count_reviews_per_item(url: str, min_reviews: int, log_every: int = 250_000) -> Counter:
    """Pass 1: count reviews per ``parent_asin`` (keeps only a counter in memory)."""
    counts: Counter = Counter()
    for i, rec in enumerate(stream_records(url), start=1):
        counts[rec["parent_asin"]] += 1
        if i % log_every == 0:
            logger.info("pass 1: %d reviews scanned, %d items", i, len(counts))
    logger.info("pass 1 done: %d reviews, %d items", sum(counts.values()), len(counts))
    return Counter({k: v for k, v in counts.items() if v >= min_reviews})


def clean_item(rec: dict[str, Any]) -> dict[str, Any]:
    """Reduce a raw metadata record to the fields CatalogIQ uses."""
    price = rec.get("price")
    try:
        price = float(price) if price not in (None, "", "None") else None
    except (TypeError, ValueError):
        price = None
    return {
        "parent_asin": rec["parent_asin"],
        "title": strip_html(rec.get("title")),
        "features": [f for f in (strip_html(x) for x in (rec.get("features") or [])) if f],
        "description": strip_html(" ".join(rec.get("description") or [])),
        "average_rating": rec.get("average_rating"),
        "rating_number": rec.get("rating_number"),
        "price": price,
        "store": strip_html(rec.get("store")),
        "categories": [strip_html(c) for c in (rec.get("categories") or [])],
        "main_category": rec.get("main_category"),
    }


def collect_meta(url: str, candidates: set[str]) -> dict[str, dict[str, Any]]:
    """Pass 2: stream item metadata, keeping only candidate items with a usable title."""
    found: dict[str, dict[str, Any]] = {}
    for i, rec in enumerate(stream_records(url), start=1):
        if rec["parent_asin"] in candidates:
            item = clean_item(rec)
            if item["title"]:
                found[item["parent_asin"]] = item
        if i % 250_000 == 0:
            logger.info("meta pass: %d records scanned, %d candidates kept", i, len(found))
    logger.info(
        "meta pass done: %d of %d candidates have usable metadata", len(found), len(candidates)
    )
    return found


def collect_reviews(
    url: str, items: set[str], max_per_item: int, min_chars: int, seed: int
) -> pd.DataFrame:
    """Pass 3: stream reviews of the selected items, clean them, cap per item (reservoir sample).

    Cleaning: strip HTML; drop reviews shorter than ``min_chars``; drop exact duplicates of the
    same text for the same item (copy-pasted/bot reviews).
    """
    rng = random.Random(seed)
    kept: dict[str, list[dict[str, Any]]] = {a: [] for a in items}
    seen_per_item: Counter = Counter()
    seen_text: set[tuple[str, str]] = set()
    dropped = Counter()
    for i, rec in enumerate(stream_records(url), start=1):
        asin = rec["parent_asin"]
        if asin not in items:
            continue
        text = strip_html(rec.get("text"))
        if len(text) < min_chars:
            dropped["too_short"] += 1
            continue
        key = (asin, text.lower())
        if key in seen_text:
            dropped["duplicate"] += 1
            continue
        seen_text.add(key)
        row = {
            "parent_asin": asin,
            "rating": rec.get("rating"),
            "title": strip_html(rec.get("title")),
            "text": text,
            "helpful_vote": rec.get("helpful_vote", 0),
            "verified_purchase": bool(rec.get("verified_purchase")),
            "timestamp": rec.get("timestamp"),
        }
        seen_per_item[asin] += 1
        n_seen = seen_per_item[asin]
        if len(kept[asin]) < max_per_item:
            kept[asin].append(row)
        else:
            j = rng.randint(0, n_seen - 1)
            if j < max_per_item:
                kept[asin][j] = row
        if i % 250_000 == 0:
            logger.info("review pass: %d records scanned", i)
    logger.info("review cleaning drops: %s", dict(dropped))
    df = pd.DataFrame([r for rows in kept.values() for r in rows])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    return df


def file_size_mb(*paths: str | Path) -> float:
    """Total size of files in megabytes."""
    return sum(Path(p).stat().st_size for p in paths) / 1e6
