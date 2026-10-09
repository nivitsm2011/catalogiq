"""Generate the example grids for manual review and the notebook.

Gallery = train+val; queries = test photos (so a query never finds its own near-duplicate twin).
Outputs: reports/figures/13_search_examples.png, 14_search_failures.png, 15_text_search.png,
         16_shop_the_look.png, reports/search_example_queries.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.config import get_search_config
from catalogiq.search.embeddings import ClipEncoder, embed_catalog
from catalogiq.search.engine import SearchEngine
from catalogiq.search.labelcheck import compute_label_check, verified_mask
from catalogiq.search.looks import ShopTheLook
from catalogiq.search.viz import plot_looks, plot_query_grid, plot_text_grid
from catalogiq.vision.data import build_image_cache

EXAMPLE_TYPES = [
    "Tshirts",
    "Casual Shoes",
    "Watches",
    "Handbags",
    "Kurtas",
    "Sunglasses",
    "Heels",
    "Dresses",
]
TEXT_QUERIES = [
    "black formal shoes for men",
    "red dress for women",
    "blue denim jeans",
    "white sports shoes",
]


def main() -> None:
    """Build grids for 8 example queries, 2 automatically found failures, text queries and looks."""
    image_emb, _, catalog = embed_catalog()
    images, _ = build_image_cache()
    images = np.asarray(images)
    split = catalog["split"].to_numpy()
    g_pos, test_pos = np.where(np.isin(split, ["train", "val"]))[0], np.where(split == "test")[0]
    gallery = SearchEngine(catalog.iloc[g_pos].reset_index(drop=True), image_emb[g_pos])
    rng = np.random.default_rng(42)

    chosen = []
    for atype in EXAMPLE_TYPES:
        cand = test_pos[(catalog.iloc[test_pos]["articleType"] == atype).to_numpy()]
        chosen.append(int(rng.choice(cand)))
    triples = []
    for p in chosen:
        res = gallery.search_vector(image_emb[p], k=5)
        triples.append((catalog.iloc[p], images[p], res))
    plot_query_grid(
        images,
        catalog,
        triples,
        "reports/figures/13_search_examples.png",
        "8 example queries (test photos) and their top-5 results",
    )
    graded = []
    for q, _, res in triples:
        for _, r in res.iterrows():
            same_type = r["articleType"] == q["articleType"]
            grade = (
                "same type + colour"
                if same_type and r["baseColour"] == q["baseColour"]
                else ("same type" if same_type else "different type")
            )
            graded.append(
                {
                    "query_id": int(q["id"]),
                    "query_type": q["articleType"],
                    "rank": int(r["rank"]),
                    "result_id": int(r["id"]),
                    "result_type": r["articleType"],
                    "result_colour": r["baseColour"],
                    "similarity": round(float(r["score"]), 4),
                    "grade": grade,
                }
            )
    pd.DataFrame(graded).to_csv("reports/search_examples_results.csv", index=False)

    # failure cases: queries whose top-5 contain at most one item of the same article type
    sample = rng.choice(test_pos, 600, replace=False)
    ranked = gallery.search_matrix(image_emb[sample], 5)
    gal_types = gallery.catalog["articleType"].to_numpy()
    same_type = (gal_types[ranked] == catalog.iloc[sample]["articleType"].to_numpy()[:, None]).sum(
        axis=1
    )
    worst = sample[np.argsort(same_type, kind="stable")][:30]
    fails = []
    for p in worst[:10]:
        res = gallery.search_vector(image_emb[p], k=5)
        fails.append((catalog.iloc[p], images[p], res))
    plot_query_grid(
        images,
        catalog,
        fails,
        "reports/figures/14_search_failures.png",
        "Candidate failure cases (top-5 with the fewest same-type results)",
    )
    rows = [
        {
            "query_id": int(q["id"]),
            "type": q["articleType"],
            "colour": q["baseColour"],
            "same_type_in_top5": int(s),
        }
        for (q, _, _), s in zip(fails, np.sort(same_type)[:10], strict=True)
    ]
    pd.DataFrame(rows).to_csv("reports/search_failure_candidates.csv", index=False)

    encoder = ClipEncoder()
    full = SearchEngine(catalog, image_emb, encoder=encoder)
    plot_text_grid(
        images,
        catalog,
        [(t, full.search_by_text(t, k=6)) for t in TEXT_QUERIES],
        "reports/figures/15_text_search.png",
    )

    check = compute_label_check()
    verified = verified_mask(
        catalog, check, get_search_config()["shop_the_look"]["verify_min_confidence"]
    )
    print(
        f"label-verified catalog items: {verified.mean():.1%} "
        f"({int(verified.sum())} of {len(verified)})"
    )
    stl = ShopTheLook(full, verified=verified)
    look_ids = []
    for atype, gender in [
        ("Shirts", "Men"),
        ("Dresses", "Women"),
        ("Jeans", "Women"),
        ("Tshirts", "Men"),
    ]:
        cand = catalog[
            (catalog["articleType"] == atype)
            & (catalog["gender"] == gender)
            & (catalog["split"] == "test")
        ]
        look_ids.append(int(cand["id"].iloc[3]))
    looks = [stl.recommend(i, per_category=2) for i in look_ids]
    plot_looks(images, catalog, looks, "reports/figures/16_shop_the_look.png")
    for look in looks:
        print(
            look.query["articleType"],
            look.query["baseColour"],
            look.query["gender"],
            look.query["usage"],
            look.query["season"],
        )
        print(
            look.items[
                ["role", "articleType", "baseColour", "usage", "season", "score", "reasons"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
