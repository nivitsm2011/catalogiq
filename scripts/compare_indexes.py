"""Compare IndexFlatIP (exact) with HNSW (approximate): speed, memory and recall.

Also simulates a larger catalog by tiling the embeddings with small noise so the crossover where an
approximate index starts to pay off is visible (the real catalog has only about 42k items).

Usage: python scripts/compare_indexes.py
Outputs: reports/index_comparison.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.config import get_search_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.search.embeddings import embed_catalog
from catalogiq.search.index import compare_indexes, l2_normalise
from catalogiq.utils import set_seed

logger = get_logger(__name__)


def main() -> None:
    """Run the comparison on the real gallery and on synthetic larger catalogs."""
    set_seed(42)
    scfg = get_search_config()
    image_emb, _, catalog = embed_catalog()
    gallery = image_emb[catalog["split"].isin(["train", "val"]).to_numpy()]
    rng = np.random.default_rng(42)
    queries = gallery[rng.choice(len(gallery), 1000, replace=False)]
    k = scfg["index"]["default_k"]

    rows = []
    for factor in (1, 5, 20):
        if factor == 1:
            emb, label = gallery, f"{len(gallery):,} items (real)"
        else:
            noise = rng.normal(scale=0.02, size=(factor - 1, *gallery.shape)).astype(np.float32)
            emb = l2_normalise(np.concatenate([gallery] + [gallery + n for n in noise]))
            label = f"{len(emb):,} items (synthetic, real vectors + noise)"
        logger.info("comparing indexes on %s", label)
        for r in compare_indexes(emb, queries, k=k):
            rows.append({"catalog": label, **r})
    table = pd.DataFrame(rows)
    table.to_csv(resolve_path("reports/index_comparison.csv"), index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
