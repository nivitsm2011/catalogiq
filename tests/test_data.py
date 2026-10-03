"""Unit tests for the data pipeline using small synthetic inputs (no downloads)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.data.audit import near_duplicate_groups
from catalogiq.data.cleaning import handle_rare_classes, normalise_colours
from catalogiq.data.fashion import load_styles
from catalogiq.data.listing_quality import add_quality_score
from catalogiq.data.reviews import strip_html
from catalogiq.data.splits import make_splits, verify_no_leakage


def test_load_styles_repairs_commas_and_counts_bad_lines(tmp_path) -> None:
    csv = tmp_path / "styles.csv"
    csv.write_text(
        "id,gender,masterCategory,subCategory,articleType,baseColour,season,year,usage,"
        "productDisplayName\n"
        "1,Men,Apparel,Topwear,Shirts,Blue,Summer,2011,Casual,Arrow Blue Shirt\n"
        "2,Men,Apparel,Topwear,Shirts,Blue,Summer,2011,Casual,Shirt, slim fit, cotton\n"
        "3,Men,Apparel,Topwear\n",
        encoding="utf-8",
    )
    df, report = load_styles(csv)
    assert len(df) == 2
    assert df.loc[1, "productDisplayName"] == "Shirt, slim fit, cotton"
    assert report.rows_repaired == 1
    assert report.rows_unrecoverable == 1
    assert report.unrecoverable_line_numbers == [4]


def test_near_duplicate_groups_respects_threshold() -> None:
    base = 0xFFFF0000FFFF0000
    hashes = pd.Series([f"{base:016x}", f"{base ^ 0b11:016x}", f"{base ^ 0xFFFF:016x}", None])
    groups = near_duplicate_groups(hashes, max_distance=2)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert groups[3] == -1


def test_group_aware_split_never_leaks() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(
        {"articleType": rng.choice(["a", "b", "c"], n), "dup_group": rng.integers(0, 600, n)}
    )
    fractions = {"train": 0.7, "val": 0.15, "test": 0.15}
    out = make_splits(df, "articleType", "dup_group", fractions, n_folds=20, seed=1)
    assert verify_no_leakage(out, "dup_group") == 0
    assert set(out["split"]) == {"train", "val", "test"}
    assert 0.6 < (out["split"] == "train").mean() < 0.8


def test_colour_normalisation_and_rare_classes() -> None:
    mapping = {"Navy Blue": "Blue", "Blue": "Blue"}
    out = normalise_colours(pd.Series(["Navy Blue", "Metallic", None]), mapping)
    assert out.tolist()[:2] == ["Blue", "Other"] and pd.isna(out.iloc[2])
    df = pd.DataFrame({"t": ["a"] * 5 + ["b"] * 1})
    merged, _ = handle_rare_classes(df, "t", min_count=3, policy="other")
    assert set(merged["t"]) == {"a", "Other"}
    dropped, info = handle_rare_classes(df, "t", min_count=3, policy="drop")
    assert set(dropped["t"]) == {"a"} and info["rows_affected"] == 1


def test_strip_html() -> None:
    assert strip_html("Great<br />fit &amp; <b>colour</b>") == "Great fit & colour"
    assert strip_html(None) == ""


def test_quality_score_prefers_well_rated_with_volume() -> None:
    items = pd.DataFrame(
        {
            "average_rating": [5.0, 4.6, 3.0],
            "rating_number": [2, 2000, 2000],
            "title": ["x" * 30] * 3,
            "features": [["a", "b", "c"]] * 3,
        }
    )
    cfg = {
        "prior_strength": 50,
        "exemplar_min_ratings": 50,
        "exemplar_min_score": 0.6,
        "exemplar_min_title_chars": 20,
        "exemplar_min_features": 3,
    }
    out = add_quality_score(items, cfg)
    assert out.loc[1, "quality_score"] > out.loc[0, "quality_score"] > 0
    assert out["is_exemplar"].tolist() == [False, True, False]


def test_clean_fashion_drops_to_fixpoint_and_merges_last() -> None:
    from catalogiq.data.cleaning import clean_fashion

    rows = (
        [("Apparel", "Topwear", "Tshirts")] * 120
        + [("Apparel", "Topwear", "Rare")] * 5
        + [("Free", "Misc", "Gift")] * 110
        + [("Free", "Misc", "Gift2")] * 3
    )
    df = pd.DataFrame(rows, columns=["masterCategory", "subCategory", "articleType"])
    df["id"] = range(len(df))
    df["image_exists"] = True
    df["image_ok"] = True
    df["baseColour"] = "Navy Blue"
    cfg = {
        "masterCategory": {"min_count": 112, "policy": "drop"},
        "subCategory": {"min_count": 5, "policy": "other"},
        "articleType": {"min_count": 10, "policy": "drop"},
        "baseColour": {"min_count": 1, "policy": "other"},
    }
    out, _ = clean_fashion(df, {"Navy Blue": "Blue"}, cfg, list(cfg))
    # Gift2 (3 rows) is dropped, which shrinks "Free" to 110 < 112, so it is dropped too (round 2).
    assert set(out["masterCategory"]) == {"Apparel"}
    assert set(out["articleType"]) == {"Tshirts"}
    assert set(out["baseColour"]) == {"Blue"}


def test_twin_disagreement() -> None:
    from catalogiq.data.target_evidence import twin_disagreement

    df = pd.DataFrame({"g": [1, 1, 2, 2, 3], "label": ["a", "a", "a", "b", "z"]})
    rate, n_groups = twin_disagreement(df, "label", "g")
    assert n_groups == 2 and rate == 0.5
