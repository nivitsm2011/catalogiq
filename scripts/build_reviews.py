"""Stream, sample and clean Amazon_Fashion reviews + item metadata (never stored in full).

Usage:  python scripts/build_reviews.py            # full run (streams ~3.5 GB, writes <300 MB)
        python scripts/build_reviews.py --limit N  # smoke test on the first N records only
"""

from __future__ import annotations

import argparse
import itertools

import pandas as pd

from catalogiq.config import get_config, resolve_path
from catalogiq.data import reviews as rv
from catalogiq.data.listing_quality import add_quality_score
from catalogiq.logging_utils import get_logger
from catalogiq.utils import set_seed, timed

logger = get_logger(__name__)


def _limited(limit: int | None):
    """Monkey-patch-free helper: wrap ``stream_records`` to stop after ``limit`` records."""
    if limit is None:
        return
    original = rv.stream_records
    rv.stream_records = lambda url: itertools.islice(original(url), limit)


@timed
def main() -> None:
    """Run the three streaming passes and write the processed parquet files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="smoke test: records per pass")
    args = parser.parse_args()
    cfg = get_config()
    set_seed(cfg["project"]["seed"])
    rcfg = cfg["data"]["reviews"]
    _limited(args.limit)
    min_rev = 1 if args.limit else rcfg["min_reviews_per_item"]

    counts = rv.count_reviews_per_item(rcfg["reviews_file"], min_rev)
    candidates = {a for a, _ in counts.most_common(rcfg["n_items"] * 2)}
    meta = rv.collect_meta(rcfg["meta_file"], candidates)
    chosen = [a for a, _ in counts.most_common() if a in meta][: rcfg["n_items"]]
    logger.info("selected %d items (min %d reviews each)", len(chosen), min_rev)

    reviews_df = rv.collect_reviews(
        rcfg["reviews_file"],
        set(chosen),
        rcfg["max_reviews_per_item"],
        rcfg["min_review_chars"],
        cfg["project"]["seed"],
    )
    items = pd.DataFrame([meta[a] for a in chosen])
    items["reviews_total_in_dump"] = items["parent_asin"].map(counts)
    items = add_quality_score(items, cfg["listing_quality"])

    rev_path, item_path = resolve_path(rcfg["reviews_parquet"]), resolve_path(rcfg["meta_parquet"])
    rev_path.parent.mkdir(parents=True, exist_ok=True)
    reviews_df.to_parquet(rev_path, index=False)
    items.to_parquet(item_path, index=False)
    size = rv.file_size_mb(rev_path, item_path)
    logger.info(
        "wrote %d reviews, %d items, %.1f MB (cap %d MB), exemplars: %d",
        len(reviews_df),
        len(items),
        size,
        rcfg["max_processed_mb"],
        int(items["is_exemplar"].sum()),
    )
    if size > rcfg["max_processed_mb"]:
        raise RuntimeError(
            f"processed output {size:.0f} MB exceeds cap; lower max_reviews_per_item"
        )


if __name__ == "__main__":
    main()
