"""Tests for the Phase 6 quality checks (no models, no data files needed)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config
from catalogiq.quality import degrade
from catalogiq.quality.completeness import completeness, is_missing, required_attributes
from catalogiq.quality.consistency import (
    clip_probabilities,
    flag_mismatch,
    inject_mismatches,
    mismatch_signals,
)
from catalogiq.quality.evaluate import per_check_scores, precision_recall
from catalogiq.quality.image_checks import check_image, corner_std, flags_from_metrics
from catalogiq.quality.score import listing_score

T = {  # fixed thresholds so the tests do not depend on tuned files
    "blur_min_laplacian_var": 300.0,
    "dark_max_mean_luma": 117.0,
    "bright_min_p5_luma": 230.0,
    "min_short_side_px": 48,
    "aspect_min": 0.55,
    "aspect_max": 1.45,
    "clutter_max_corner_std": 37.0,
    "text_min_clip_margin": 1.5,
}


def _product(h: int = 80, w: int = 60) -> np.ndarray:
    """A plain white photo with a sharp striped dark product in the middle."""
    img = np.full((h, w, 3), 255, np.uint8)
    for k in range(0, 36, 6):
        img[20 + k : 23 + k, 15:45] = (20, 20, 20)
        img[23 + k : 26 + k, 15:45] = (200, 40, 40)
    return img


# ------------------------------------------------------------------ image checks
def test_clean_photo_raises_nothing() -> None:
    res = check_image(_product(), T)
    assert res["issues"] == []


def test_blur_and_darkness_are_flagged() -> None:
    assert "blurry" in check_image(degrade.blur(_product(), 2.0), T)["issues"]
    assert "too_dark" in check_image(degrade.darken(_product(), 0.3), T)["issues"]


def test_resolution_and_aspect_rules() -> None:
    assert "low_resolution" in check_image(degrade.downscale(_product(), 0.4), T)["issues"]
    assert "bad_aspect_ratio" in check_image(degrade.stretch(_product(), 2.0), T)["issues"]
    assert "bad_aspect_ratio" not in check_image(_product(80, 60), T)["issues"]


def test_clutter_raises_corner_spread() -> None:
    rng = np.random.default_rng(0)
    cluttered = degrade.clutter(_product(), rng)
    assert corner_std(_product(), 0.2) == 0.0
    assert corner_std(cluttered, 0.2) > 37.0
    assert "cluttered_background" in check_image(cluttered, T)["issues"]


def test_text_check_needs_the_clip_score() -> None:
    m = {
        "blur": 1e4,
        "mean_luma": 200,
        "p5_luma": 50,
        "short_side": 60,
        "aspect": 0.75,
        "corner_std": 0.0,
        "text_chars": 0,
    }
    assert not flags_from_metrics(m, T)["text_or_watermark"]
    assert flags_from_metrics(m, T, text_margin=5.0)["text_or_watermark"]
    assert not flags_from_metrics(m, T, text_margin=0.0)["text_or_watermark"]


def test_degradations_keep_valid_images() -> None:
    rng = np.random.default_rng(1)
    img = _product()
    assert degrade.stretch(img, 1.8).shape[:2] == (80, 144)
    assert degrade.downscale(img, 0.5).shape[:2] == (40, 30)
    assert degrade.overlay_text(img, rng).shape == img.shape
    types = {t for _, t, _, _ in degrade.make_degraded_set(img[None], rng)}
    assert types == {"blur", "dark", "bright", "lowres", "aspect", "clutter", "text"}


def test_precision_recall_and_per_check_scoring() -> None:
    res = precision_recall(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert res["precision"] == 0.5 and res["recall"] == 0.5 and res["tp"] == 1
    checks = [
        "blurry",
        "too_dark",
        "overexposed",
        "low_resolution",
        "bad_aspect_ratio",
        "cluttered_background",
        "text_or_watermark",
    ]
    df = pd.DataFrame(
        {
            "kind": ["clean", "degraded", "degraded"],
            "type": ["", "blur", "blur"],
            "level": [-1, 0, 2],
        }
    )
    for c in checks:
        df[c] = False
    df.loc[2, "blurry"] = True  # level 2 caught; level 0 is below positive_from_level and ignored
    scores = per_check_scores(df)
    assert scores.loc["blurry", "recall"] == 1.0 and scores.loc["blurry", "precision"] == 1.0


# -------------------------------------------------------------------- completeness
def test_missing_values_and_required_attributes() -> None:
    assert (
        is_missing(None) and is_missing(float("nan")) and is_missing("  ") and is_missing("Unknown")
    )
    assert not is_missing("Blue")
    group, required = required_attributes("Watches")
    assert group == "watches" and "baseColour" in required and "productDisplayName" in required


def test_completeness_percentage() -> None:
    full = {
        "gender": "Men",
        "articleType": "Watches",
        "baseColour": "Black",
        "productDisplayName": "X",
    }
    assert completeness(full)["pct"] == 100.0
    partial = {**full, "baseColour": None}
    res = completeness(partial)
    assert res["missing"] == ["baseColour"] and 0 < res["pct"] < 100


# ----------------------------------------------------------------------------- score
def test_score_weights_and_fixes() -> None:
    cfg = get_quality_config()["score"]
    clean = listing_score([], [], 100.0, [])
    assert clean["score"] == 100.0 and clean["flags"] == [] and not clean["needs_attention"]
    bad = listing_score(["blurry"], [], 100.0, [])
    expected = 100 - cfg["weights"]["image"] * cfg["image_penalty"]["blurry"] / sum(
        cfg["weights"].values()
    )
    assert bad["score"] == round(expected, 1)
    assert "Retake the photo" in bad["flags"][0]["fix"]
    mism = listing_score(
        [], [{"attribute": "baseColour", "stated": "Blue", "seen": "Red"}], 100.0, []
    )
    assert "listed Blue" in mism["flags"][0]["fix"] and mism["components"]["consistency"] < 100
    floor = listing_score(list(get_quality_config()["score"]["image_penalty"]), [], 0.0, ["gender"])
    assert floor["components"]["image"] == 0.0 and 0.0 <= floor["score"] < 100.0
    assert floor["needs_attention"]
    assert {f["area"] for f in floor["flags"]} == {"image", "completeness"}


# ------------------------------------------------------------------------ consistency
def test_clip_probabilities_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    img = rng.normal(size=(5, 8)).astype(np.float32)
    img /= np.linalg.norm(img, axis=1, keepdims=True)
    txt = rng.normal(size=(3, 8)).astype(np.float32)
    txt /= np.linalg.norm(txt, axis=1, keepdims=True)
    assert np.allclose(clip_probabilities(img, txt).sum(axis=1), 1.0)


def test_mismatch_signals_and_flag_modes() -> None:
    stated = np.array([0, 0])
    clf = np.array([[0.05, 0.9, 0.05], [0.9, 0.05, 0.05]])  # listing 0: classifier says class 1
    clip = np.array([[0.2, 0.7, 0.1], [0.6, 0.2, 0.2]])
    sig = mismatch_signals(stated, clf, clip)
    assert sig["clf_alt_conf"].tolist() == [0.9, 0.0]
    assert np.isclose(sig["clip_margin"].iat[0], 0.5) and sig["clip_margin"].iat[1] < 0
    assert flag_mismatch(sig, 0.8, 0.4, "clf").tolist() == [True, False]
    assert flag_mismatch(sig, 0.95, 0.4, "or").tolist() == [True, False]
    assert flag_mismatch(sig, 0.95, 0.4, "and").tolist() == [False, False]


def test_injection_counts_and_rules() -> None:
    n = 400
    rng = np.random.default_rng(0)
    cat = pd.DataFrame(
        {
            "baseColour": rng.choice(["Blue", "Red", "Green", "Black", "Grey", "Beige"], n),
            "articleType": rng.choice(["Shirts", "Jeans", "Watches", "Heels"], n),
            "gender": rng.choice(["Men", "Women", "Unisex"], n),
        }
    )
    cat["subCategory"] = cat["articleType"].map(
        {"Shirts": "Topwear", "Jeans": "Bottomwear", "Watches": "Watches", "Heels": "Shoes"}
    )
    labels = {
        "baseColour": ["Blue", "Red", "Green", "Black", "Grey", "Beige"],
        "articleType": ["Shirts", "Jeans", "Watches", "Heels"],
        "gender": ["Men", "Women", "Unisex"],
    }
    out = inject_mismatches(cat, labels, 20, np.random.default_rng(1))
    for a in labels:
        inj = out[f"injected_{a}"]
        assert inj.sum() == 20
        assert (out.loc[inj, f"stated_{a}"] != out.loc[inj, a]).all()
        assert (out.loc[~inj, f"stated_{a}"] == out.loc[~inj, a]).all()
    total = sum(out[f"injected_{a}"].astype(int) for a in labels)
    assert total.max() == 1  # at most one injected mismatch per listing
    gender = out[out["injected_gender"]]
    assert (
        set(gender["gender"]) <= {"Men", "Women"}
        and (gender["stated_gender"] != gender["gender"]).all()
    )
    types = out[out["injected_articleType"]]
    assert (
        types["stated_articleType"].map(
            cat.drop_duplicates("articleType").set_index("articleType")["subCategory"]
        )
        != types["subCategory"]
    ).all()
