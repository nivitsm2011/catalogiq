"""Embed every catalog image and attribute sentence with OpenCLIP (resumable, cached).

Usage: python scripts/embed_catalog.py [--force]
Outputs: data/processed/clip_image_emb.npy, clip_text_emb.npy, clip_ids.npy, search_catalog.parquet,
         data/processed/clf_image_emb.npy (Phase 2 classifier features, used as a baseline)
"""

from __future__ import annotations

import argparse

from catalogiq.logging_utils import get_logger
from catalogiq.search.embeddings import classifier_embeddings, embed_catalog
from catalogiq.utils import set_seed, timed

logger = get_logger(__name__)


@timed
def main() -> None:
    """Compute (or load) all embeddings."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    clf = classifier_embeddings(force=args.force)
    logger.info("classifier features: %s", clf.shape)
    image_emb, text_emb, catalog = embed_catalog(force=args.force)
    logger.info(
        "CLIP image %s, text %s, catalog %s", image_emb.shape, text_emb.shape, catalog.shape
    )


if __name__ == "__main__":
    main()
