"""Data cleaning on the committed fixtures: styles, image audit, review sampling, text stats."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from catalogiq.config import PROJECT_ROOT
from catalogiq.data import reviews as reviews_mod
from catalogiq.data.audit import audit_summary, inspect_images, near_duplicate_groups
from catalogiq.data.cleaning import clean_fashion, load_colour_map
from catalogiq.data.fashion import image_path, load_styles
from catalogiq.data.reviews import clean_item, collect_meta, collect_reviews, file_size_mb
from catalogiq.data.target_evidence import target_evidence, twin_disagreement
from catalogiq.data.text_stats import listing_profile, topic_share

TAXONOMY = ["masterCategory", "subCategory", "articleType", "baseColour", "gender"]


@pytest.fixture()
def styles(fixtures_dir: Path):
    df, report = load_styles(fixtures_dir / "styles_sample.csv")
    df["id"] = df["id"].astype(int)
    return df, report


@pytest.fixture()
def image_dir(tmp_path: Path, photo_dir: Path) -> Path:
    """Photos named <id>.jpg for most products; one id has no file, one is not an image."""
    out = tmp_path / "images"
    out.mkdir()
    for pid in (1001, 1002, 1003, 1004, 1005, 1007):
        shutil.copy(photo_dir / "shirt_blue.jpg", out / f"{pid}.jpg")
    (out / "1006.jpg").write_bytes(b"corrupt")  # exists but cannot be decoded
    return out


def test_styles_sample_is_repaired_and_one_line_is_unrecoverable(styles) -> None:
    df, report = styles
    assert len(df) == 7 and report.rows_repaired == 1 and report.rows_unrecoverable == 1
    assert df.loc[df["id"] == 1002, "productDisplayName"].item() == "Shirt, slim fit, cotton"


def test_audit_flags_missing_unreadable_and_near_duplicate_images(styles, image_dir: Path) -> None:
    df, report = styles
    audited = inspect_images(df, image_dir, min_side=20, workers=2)
    by_id = audited.set_index("id")
    assert by_id.loc[1001, "image_ok"] and by_id.loc[1001, "phash"]
    assert not by_id.loc[1006, "image_ok"] and by_id.loc[1006, "image_exists"]  # corrupt file
    assert image_path(image_dir, 1008) == image_dir / "1008.jpg"
    # the same drawing under different ids collapses into one near-duplicate group
    audited["dup_group"] = near_duplicate_groups(audited["phash"], max_distance=1)
    assert audited.loc[audited["id"].isin([1001, 1002, 1003]), "dup_group"].nunique() == 1
    summary = audit_summary(audited, ["articleType"], report.as_dict())
    assert summary["rows"] == 7 and summary["image_unreadable_or_tiny"] == 1
    assert summary["near_duplicate_groups_size_ge2"] >= 1
    assert json.dumps(summary)  # JSON-serialisable for reports/fashion_audit.json


def test_missing_image_file_is_reported_not_raised(styles, tmp_path: Path) -> None:
    df, _ = styles
    audited = inspect_images(df.head(2), tmp_path, min_side=20, workers=1)
    assert not audited["image_exists"].any() and audited["phash"].isna().all()


def test_clean_fashion_logs_every_step_and_normalises_colours(styles, image_dir: Path) -> None:
    df, report = styles
    audited = inspect_images(df, image_dir, min_side=20, workers=2)
    audited["dup_group"] = -1
    colour_map = load_colour_map(PROJECT_ROOT / "configs" / "colour_map.yaml")
    rare = {t: {"min_count": 1, "policy": "other"} for t in TAXONOMY}
    clean, log = clean_fashion(audited, colour_map, rare, TAXONOMY)
    assert [s["step"] for s in log][0] == "drop duplicate ids"
    assert 1006 not in set(clean["id"]) and len(clean) <= len(audited)
    assert clean["baseColour"].isin(set(colour_map.values()) | {"Other"}).all()
    assert all(s["rows_after"] <= s["rows_before"] for s in log[:3])


def test_twin_disagreement_and_target_evidence() -> None:
    df = pd.DataFrame(
        {
            "grp": [1, 1, 2, 2, 3],
            "articleType": ["Shirts", "Shirts", "Tops", "Dress", "Watches"],
            "baseColour": ["Blue", "Blue", "Red", "Red", "Black"],
        }
    )
    rate, pairs = twin_disagreement(df, "articleType", "grp")
    assert pairs == 2 and rate == pytest.approx(0.5)
    raw = df.assign(articleType=df["articleType"].where(df["grp"] != 3))  # one missing label
    evidence = target_evidence(raw, df, ["articleType", "baseColour"], "grp").set_index("target")
    assert evidence.loc["articleType", "pct_missing"] == pytest.approx(20.0)
    assert evidence.loc["baseColour", "twin_disagreement"] == pytest.approx(0.0)
    assert evidence.loc["articleType", "n_classes"] == 4


# ------------------------------------------------------------------------- reviews pipeline
@pytest.fixture()
def fake_stream(monkeypatch: pytest.MonkeyPatch, fixtures_dir: Path):
    """Serve the JSONL fixtures instead of streaming from Hugging Face (no network)."""

    def stream(url: str):
        path = fixtures_dir / ("reviews_sample.jsonl" if "review" in url else "meta_sample.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)

    monkeypatch.setattr(reviews_mod, "stream_records", stream)


def test_count_reviews_keeps_only_items_with_enough_reviews(fake_stream) -> None:
    counts = reviews_mod.count_reviews_per_item("reviews", min_reviews=3, log_every=2)
    assert dict(counts) == {"B0AAA": 4}


def test_clean_item_strips_html_and_parses_price(fixtures_dir: Path) -> None:
    rows = [json.loads(x) for x in (fixtures_dir / "meta_sample.jsonl").read_text().splitlines()]
    a, b, c = (clean_item(r) for r in rows)
    assert a["title"] == "Floral Summer Dress" and a["features"] == [
        "Soft fabric",
        "Machine washable",
    ]
    assert (
        a["price"] == pytest.approx(29.99)
        and a["description"] == "A light dress. Great for summer."
    )
    assert b["price"] is None and b["features"] == [] and b["store"] == ""
    assert c["price"] == 12.0


def test_collect_meta_drops_items_without_a_title(fake_stream) -> None:
    found = collect_meta("meta", {"B0AAA", "B0BBB", "B0DDD"})
    assert set(found) == {"B0AAA", "B0BBB"}  # B0DDD has an empty title


def test_collect_reviews_drops_short_duplicate_and_other_items(fake_stream) -> None:
    df = collect_reviews("reviews", {"B0AAA", "B0BBB"}, max_per_item=10, min_chars=30, seed=1)
    assert set(df["parent_asin"]) == {"B0AAA", "B0BBB"}
    assert len(df) == 4  # the 8-character review and the exact duplicate were dropped
    assert not df["text"].str.contains("<b>").any()
    assert str(df["timestamp"].dtype).startswith("datetime64")


def test_review_cap_per_item_keeps_a_reservoir_sample(fake_stream) -> None:
    a = collect_reviews("reviews", {"B0AAA"}, max_per_item=1, min_chars=30, seed=7)
    b = collect_reviews("reviews", {"B0AAA"}, max_per_item=1, min_chars=30, seed=7)
    assert len(a) == 1 and a["text"].tolist() == b["text"].tolist()  # fixed seed: repeatable


def test_file_size_mb(tmp_path: Path) -> None:
    f = tmp_path / "x.bin"
    f.write_bytes(b"0" * 2_000_000)
    assert file_size_mb(f) == pytest.approx(2.0)


# ------------------------------------------------------------------------------ text stats
def test_listing_profile_counts_copy_and_attribute_mentions() -> None:
    items = pd.DataFrame(
        {
            "title": ["Blue cotton shirt for men", "Bag"],
            "features": [["Machine washable", "Slim fit"], []],
            "description": ["Soft and comfortable.", ""],
        }
    )
    prof = listing_profile(items)
    assert prof["n_bullets"].tolist() == [2, 0] and prof["title_chars"].iloc[1] == 3
    assert prof["n_attribute_families"].iloc[0] > prof["n_attribute_families"].iloc[1]


def test_topic_share_counts_whole_words() -> None:
    reviews = pd.DataFrame(
        {
            "title": ["", "Nice", None],
            "text": ["too small, poor fit", "great value for the price", "fell apart"],
        }
    )
    share = topic_share(reviews, {"size": ["small", "fit"], "price": ["price", "value"]})
    assert share["size"] == pytest.approx(1 / 3) and share["price"] == pytest.approx(1 / 3)
