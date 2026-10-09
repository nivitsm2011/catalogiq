"""Tests for the search features using small synthetic catalogs and a fake encoder (no CLIP)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catalogiq.search.duplicates import (
    DuplicateFinder,
    duplicate_groups,
    evaluate_thresholds,
    hamming_hex,
)
from catalogiq.search.embeddings import build_text
from catalogiq.search.engine import SearchEngine
from catalogiq.search.index import compare_indexes, l2_normalise
from catalogiq.search.labelcheck import verified_mask
from catalogiq.search.looks import ShopTheLook, attributes_from_predictions
from catalogiq.search.metrics import (
    average_precision_at_k,
    count_relevant,
    precision_at_k,
    recall_at_k,
    relevance_matrix,
)

DIM = 16


def make_catalog() -> tuple[pd.DataFrame, np.ndarray]:
    """Two clusters per article type; shirts/watches/shoes/jeans/bags with colours and genders."""
    rng = np.random.default_rng(0)
    specs = [
        ("Shirts", "Blue", "Men", "Casual", "Summer", 12),
        ("Shirts", "Red", "Men", "Casual", "Summer", 12),
        ("Jeans", "Blue", "Men", "Casual", "Summer", 12),
        ("Casual Shoes", "White", "Men", "Casual", "Summer", 12),
        ("Watches", "Black", "Men", "Casual", "Summer", 12),
        ("Handbags", "Black", "Women", "Casual", "Summer", 12),
        ("Shirts", "Blue", "Women", "Casual", "Summer", 12),
        ("Formal Shoes", "Black", "Men", "Formal", "Winter", 12),
        ("Briefs", "Pink", "Women", "Casual", "Summer", 6),
        ("Trousers", "Black", "Men", "Formal", "Winter", 12),
        ("Sports Shoes", "White", "Men", "Sports", "Summer", 12),
        ("Flip Flops", "Black", "Men", "Casual", "Summer", 6),
    ]
    centers = rng.normal(size=(len(specs), DIM))
    rows, vecs = [], []
    for ci, (atype, colour, gender, usage, season, n) in enumerate(specs):
        for _ in range(n):
            rows.append(
                {
                    "id": 1000 + len(rows),
                    "articleType": atype,
                    "baseColour": colour,
                    "gender": gender,
                    "usage": usage,
                    "season": season,
                    "masterCategory": "Apparel",
                    "subCategory": "Topwear",
                    "productDisplayName": f"{colour} {atype}",
                    "dup_group": len(rows),
                }
            )
            vecs.append(centers[ci] + 0.05 * rng.normal(size=DIM))
    return pd.DataFrame(rows), np.array(vecs, dtype=np.float32)


class FakeEncoder:
    """Maps an 'image' (here: a catalog row index) or text to a vector of the synthetic space."""

    def __init__(self, emb: np.ndarray, catalog: pd.DataFrame) -> None:
        self.emb, self.catalog = emb, catalog

    def encode_image(self, image):  # image is a row position for the fake
        return self.emb[int(image)] if np.isscalar(image) else self.emb[0]

    def encode_text(self, text: str):
        # the first catalog row whose article type appears in the text
        for i, a in enumerate(self.catalog["articleType"]):
            if a.lower() in text.lower():
                return self.emb[i]
        return self.emb[0]


@pytest.fixture(scope="module")
def engine() -> SearchEngine:
    catalog, emb = make_catalog()
    return SearchEngine(catalog, emb, encoder=FakeEncoder(emb, catalog))


def test_search_by_id_returns_k_sorted_and_excludes_self(engine: SearchEngine) -> None:
    qid = int(engine.catalog["id"][0])
    res = engine.search_by_id(qid, k=5)
    assert len(res) == 5 and qid not in set(res["id"])
    assert (np.diff(res["score"].to_numpy()) <= 1e-6).all()
    assert (res["articleType"] == "Shirts").all()
    assert list(res["rank"]) == [1, 2, 3, 4, 5]


def test_search_by_image_matches_cached_neighbours(engine: SearchEngine) -> None:
    res = engine.search_by_image(30, k=6)  # row 30 is a Jeans item
    assert (res["articleType"] == "Jeans").sum() >= 5
    assert res["score"].iloc[0] == pytest.approx(1.0, abs=1e-3)


def test_filters_restrict_and_never_return_fewer_than_available(engine: SearchEngine) -> None:
    res = engine.search_by_id(
        int(engine.catalog["id"][0]), k=10, filters={"gender": "Women", "articleType": "Shirts"}
    )
    assert len(res) == 10 and (res["gender"] == "Women").all()
    multi = engine.search_by_image(0, k=50, filters={"baseColour": ["Blue", "Red"]})
    assert set(multi["baseColour"]) <= {"Blue", "Red"}
    assert len(engine.search_by_image(0, k=5, filters={"articleType": "Nonexistent"})) == 0
    with pytest.raises(ValueError):
        engine.search_by_image(0, filters={"price": "10"})


def test_search_by_text_and_validation(engine: SearchEngine) -> None:
    res = engine.search_by_text("blue watches for men", k=5)
    assert (res["articleType"] == "Watches").all()
    with pytest.raises(ValueError):
        engine.search_by_text("   ")
    no_encoder = SearchEngine(engine.catalog, engine.emb)
    with pytest.raises(RuntimeError):
        no_encoder.search_by_text("shirt")


def test_hnsw_index_agrees_with_flat() -> None:
    catalog, emb = make_catalog()
    flat = SearchEngine(catalog, emb, index_kind="flat")
    hnsw = SearchEngine(catalog, emb, index_kind="hnsw")
    qid = int(catalog["id"][5])
    a, b = flat.search_by_id(qid, k=8), hnsw.search_by_id(qid, k=8)
    assert len(set(a["id"]) & set(b["id"])) >= 7
    rows = compare_indexes(emb, emb[:20], k=5, ef_search_values=(16,))
    assert rows[0]["recall@5_vs_flat"] == 1.0 and rows[1]["recall@5_vs_flat"] > 0.8


def test_metrics_known_values() -> None:
    q = pd.DataFrame({"articleType": ["a", "b"], "baseColour": ["x", "x"]})
    g = pd.DataFrame({"articleType": ["a", "a", "b", "b"], "baseColour": ["x", "y", "x", "x"]})
    ranked = np.array([[1, 0, 2], [3, 2, 0]])
    rel = relevance_matrix(q, g, ranked, ["articleType", "baseColour"])
    assert rel.tolist() == [[False, True, False], [True, True, False]]
    assert recall_at_k(rel, 1) == 0.5 and recall_at_k(rel, 3) == 1.0
    assert precision_at_k(rel, 3) == pytest.approx((1 / 3 + 2 / 3) / 2)
    n_rel = count_relevant(q, g, ["articleType", "baseColour"])
    assert n_rel.tolist() == [1, 2]
    # query 1: AP = (P@2 * 1) / min(3, 1) = 0.5 ; query 2: AP = (1*1 + 1*1) / min(3, 2) = 1.0
    assert average_precision_at_k(rel, n_rel, 3) == pytest.approx(0.75)


def test_shop_the_look_rules(engine: SearchEngine) -> None:
    look = ShopTheLook(engine).recommend(int(engine.catalog["id"][0]))  # blue men's casual shirt
    assert look.query["role"] == "top"
    items = look.items
    assert not items.empty
    assert "top" not in set(items["role"])  # never the same role as the query
    assert set(items["role"]) <= {"bottom", "footwear", "accessory", "bag"}
    assert not items["gender"].isin(["Women"]).any()  # gender-compatible only
    assert not (items["articleType"] == "Briefs").any()  # underwear is never recommended
    assert items["reasons"].str.contains("visual style fit").all()
    assert int(engine.catalog["id"][0]) not in set(items["id"])
    # casual query -> casual shoes should outrank formal shoes in the footwear slot
    full = ShopTheLook(engine).recommend(int(engine.catalog["id"][0]), per_category=30).items
    shoes = full[full["role"] == "footwear"].reset_index(drop=True)
    first = {a: shoes.index[shoes["articleType"] == a].min() for a in shoes["articleType"].unique()}
    assert first["Casual Shoes"] < first["Formal Shoes"]
    assert shoes.loc[first["Casual Shoes"], "usage_score"] == 1.0
    assert "Sports Shoes" not in first  # sporty only: shares no style with a shirt


def test_shop_the_look_scores_and_non_outfit_query(engine: SearchEngine) -> None:
    stl = ShopTheLook(engine)
    colours = pd.Series(["Black", "Blue", "Red", "Multi", "Other"])
    assert stl.colour_score("Blue", colours).tolist() == [1.0, 0.7, 0.2, 0.3, 0.5]
    assert stl.colour_score("Blue", pd.Series(["White", "Beige"])).tolist() == [1.0, 1.0]
    seasons = pd.Series(["Summer", "Fall", "Winter", "nan"])
    assert stl.season_score("Summer", seasons).tolist() == [1.0, 0.5, 0.0, 0.5]
    assert stl.usage_score("Casual", pd.Series(["Casual", "Formal", "Other"])).tolist() == [
        1,
        0,
        0.5,
    ]
    briefs_id = int(engine.catalog.loc[engine.catalog["articleType"] == "Briefs", "id"].iloc[0])
    result = stl.recommend(briefs_id)
    assert result.query["role"] is None and result.items.empty


def test_shop_the_look_photo_query_uses_attribute_fn(engine: SearchEngine) -> None:
    preds = {
        "articleType": ("Shirts", 0.9, False),
        "baseColour": ("Blue", 0.8, False),
        "gender": ("Men", 0.99, False),
        "usage": ("Casual", 0.95, False),
        "season": ("Summer", 0.7, False),
    }
    stl = ShopTheLook(engine, attribute_fn=lambda img: attributes_from_predictions(preds))
    look = stl.recommend(int(engine.catalog["id"][0]))  # catalog id path
    from_photo = ShopTheLook(engine, attribute_fn=lambda img: attributes_from_predictions(preds))
    photo_look = from_photo.recommend(np.zeros((4, 4, 3)))  # not an int -> treated as a photo
    assert photo_look.query["role"] == "top" and not photo_look.items.empty
    assert look is not None
    with pytest.raises(RuntimeError):
        ShopTheLook(engine).recommend(np.zeros((4, 4, 3)))


def test_duplicate_finder_threshold_and_phash() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(6, DIM)).astype(np.float32)
    emb = np.vstack([base, base[:2] + 0.001 * rng.normal(size=(2, DIM))])  # rows 6,7 duplicate 0,1
    catalog = pd.DataFrame({"id": range(100, 108)})
    hashes = pd.Series(
        ["ffff0000ffff0000"] * 2
        + ["0000ffff0000ffff"] * 4
        + ["ffff0000ffff0001", "ffff0000ffff0000"],
        index=catalog["id"],
    )
    finder = DuplicateFinder(catalog, emb, hashes, candidate_k=3, min_similarity=0.5)
    pairs = finder.find_duplicates(0.99, max_phash_distance=4)
    assert {frozenset(p) for p in zip(pairs["id_a"], pairs["id_b"], strict=True)} == {
        frozenset((100, 106)),
        frozenset((101, 107)),
    }
    # a hash that disagrees vetoes the pair even though the embedding says duplicate
    vetoed_hashes = hashes.copy()
    vetoed_hashes.loc[106] = "0000ffff0000ffff"
    vetoed = DuplicateFinder(
        catalog, emb, vetoed_hashes, candidate_k=3, min_similarity=0.5
    ).find_duplicates(0.99, max_phash_distance=4)
    assert len(vetoed) == 1
    assert len(finder.find_duplicates(0.99, require_phash=False)) == 2
    groups = duplicate_groups(pairs)
    assert groups[100] == groups[106] != groups[101]


def test_hamming_and_threshold_evaluation() -> None:
    assert hamming_hex("0f", "00") == 4 and np.isnan(hamming_hex(None, "00"))
    labelled = pd.DataFrame(
        {
            "similarity": [0.99, 0.97, 0.93, 0.90, 0.85],
            "phash_distance": [0, 2, 20, 3, 30],
            "label": [1, 1, 0, 1, 0],
        }
    )
    table = evaluate_thresholds(labelled, [0.9, 0.95], max_phash_distance=8)
    row = table[(table["method"] == "similarity only") & (table["threshold"] == 0.95)].iloc[0]
    assert row["precision"] == 1.0 and row["recall"] == pytest.approx(2 / 3, abs=1e-3)
    low = table[(table["method"] == "similarity + pHash") & (table["threshold"] == 0.9)].iloc[0]
    assert low["precision"] == 1.0 and low["recall"] == 1.0  # pHash vetoes the 0.93 false positive


def test_build_text_drops_missing_attributes() -> None:
    row = pd.Series(
        {
            "baseColour": "Navy",
            "articleType": "Shirts",
            "gender": "Men",
            "usage": None,
            "season": np.nan,
        }
    )
    assert build_text(row) == "a product photo of a navy shirts for men"
    full = pd.Series(
        {
            "baseColour": "Other",
            "articleType": "Watches",
            "gender": "Women",
            "usage": "Formal",
            "season": "Fall",
        }
    )
    assert build_text(full) == "a product photo of a watches for women, formal style, fall season"


def test_l2_normalise() -> None:
    x = l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert x[0].tolist() == pytest.approx([0.6, 0.8]) and np.isfinite(x).all()


def test_style_family_is_a_hard_rule(engine: SearchEngine) -> None:
    trousers_id = int(engine.catalog.loc[engine.catalog["articleType"] == "Trousers", "id"].iloc[0])
    look = ShopTheLook(engine).recommend(trousers_id, per_category=5)
    shoes = look.items[look.items["role"] == "footwear"]
    assert set(shoes["articleType"]) == {"Formal Shoes"}  # casual/sports shoes share no style
    assert "both formal" in shoes.iloc[0]["reasons"]
    stl = ShopTheLook(engine)
    scores = stl.pairing_score("Trousers", pd.Series(["Formal Shoes", "Sports Shoes"])).tolist()
    assert scores[0] > 0.0 and scores[1] == 0.0
    assert stl.pairing_score("Unknown Type", pd.Series(["Jeans"])).tolist() == [0.5]


def test_label_verified_gate_blocks_unconfirmed_items(engine: SearchEngine) -> None:
    shirt_id = int(engine.catalog["id"][0])
    verified = (engine.catalog["articleType"] != "Casual Shoes").to_numpy()
    gated = ShopTheLook(engine, verified=verified).recommend(shirt_id, per_category=20)
    assert "Casual Shoes" not in set(gated.items["articleType"])
    assert "Formal Shoes" in set(gated.items["articleType"])
    no_gate = ShopTheLook(engine).recommend(shirt_id, per_category=20)
    assert "Casual Shoes" in set(no_gate.items["articleType"])


def test_verified_mask_uses_agreement_and_confidence() -> None:
    catalog = pd.DataFrame({"id": [1, 2, 3, 4], "articleType": ["A", "A", "B", "C"]})
    check = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "pred_articleType": ["A", "A", "A"],
            "pred_confidence": [0.9, 0.3, 0.9],
        }
    )
    assert verified_mask(catalog, check, 0.5).tolist() == [True, False, False, False]
