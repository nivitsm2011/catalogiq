"""More unit tests: aspect tagging, style guide, review chunking, vision data/evaluation/model."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from catalogiq.config import get_qa_config, get_rag_config
from catalogiq.rag import aspects, reviews_index, style_guide
from catalogiq.search.labelcheck import verified_mask
from catalogiq.vision import data as vdata
from catalogiq.vision.calibration import apply_threshold
from catalogiq.vision.evaluate import (
    confusion_top_k,
    per_class_table,
    reliability_bins,
)
from catalogiq.vision.model import MultiHeadNet

# ------------------------------------------------------------------------------ aspect tagging


def test_rule_tag_finds_keywords_per_aspect() -> None:
    names = aspects.aspect_names()
    texts = ["Runs small, order a size up", "The fabric feels cheap", "Nothing relevant here"]
    tags = aspects.rule_tag(texts)
    assert tags.shape == (3, len(names)) and tags.dtype == bool
    assert tags[0].any() and tags[1].any() and not tags[2].any()
    hits = aspects.rule_matches(texts[0])
    assert set(hits) == set(names) and any(hits.values())


def test_prf_micro_macro_and_best_threshold() -> None:
    n = len(aspects.aspect_names())
    y_true = np.zeros((4, n), dtype=bool)
    y_true[0, 0] = y_true[1, 0] = y_true[2, 1] = True
    perfect = aspects.prf(y_true, y_true)
    assert perfect.loc[perfect["aspect"] == "MICRO", "f1"].item() == pytest.approx(1.0)
    none = aspects.prf(y_true, np.zeros_like(y_true))
    assert none.loc[none["aspect"] == "MICRO", "recall"].item() == 0.0
    assert list(perfect["aspect"])[-2:] == ["MICRO", "MACRO"]
    scores = np.where(y_true, 0.9, 0.1)
    assert aspects.best_threshold(scores, y_true, [0.05, 0.5, 0.95]) == 0.5


def test_llm_tag_survives_a_bad_reply() -> None:
    names = aspects.aspect_names()
    replies = [json.dumps({names[0]: True}), "not json"]
    llm = SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(text=replies.pop(0)))
    out = aspects.llm_tag(["first review", "second review"], llm)
    assert out[0, 0] and out[0].sum() == 1 and not out[1].any()  # the bad reply is all-False


def test_tag_with_dispatches_and_rejects_unknown_methods(monkeypatch: pytest.MonkeyPatch) -> None:
    n = len(aspects.aspect_names())
    monkeypatch.setattr(aspects, "embedding_scores", lambda texts: np.full((len(texts), n), 0.9))
    monkeypatch.setattr(aspects, "nli_scores", lambda texts: np.full((len(texts), n), 0.1))
    assert aspects.tag_with("embedding", ["x"], threshold=0.5).all()
    assert not aspects.tag_with("nli", ["x"], threshold=0.5).any()
    assert aspects.tag_with("rules", ["too small"]).shape == (1, n)
    with pytest.raises(ValueError, match="unknown aspect method"):
        aspects.tag_with("magic", ["x"])
    result, seconds = aspects.timed(lambda: 7)
    assert result == 7 and seconds >= 0
    assert get_qa_config()["aspects"]["names"]


# ----------------------------------------------------------------------------------- style guide
def test_style_guide_text_comes_from_the_config() -> None:
    cfg = get_rag_config()
    text = style_guide.global_rules()
    assert str(cfg["style"]["bullets"]) in text and "Banned" in text
    group = next(iter(cfg["categories"]["groups"]))
    assert group in style_guide.group_rules(group)
    chunks = style_guide.style_chunks()
    assert chunks[0]["id"] == "style_all" and len(chunks) == 1 + len(cfg["categories"]["groups"])
    assert len({c["id"] for c in chunks}) == len(chunks)
    md = style_guide.render_markdown()
    assert md.startswith("# CatalogIQ category style guide") and "| Category |" in md


def test_write_style_guide_writes_the_configured_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(get_rag_config()["paths"], "style_guide_md", str(tmp_path / "guide.md"))
    path = style_guide.write_style_guide()
    assert Path(path).read_text(encoding="utf-8").startswith("# CatalogIQ category style guide")


# ------------------------------------------------------------------------------ review chunking
def test_short_reviews_stay_whole_and_long_ones_become_overlapping_windows() -> None:
    short = reviews_index.chunk_review("Nice", "Fits well.", 200, 3, 2, 500, True)
    assert short == ["Nice. Fits well."]
    text = " ".join(f"Sentence number {i} is here." for i in range(8))
    chunks = reviews_index.chunk_review("Title", text, 20, 3, 2, 500, True)
    assert len(chunks) > 1 and chunks[0].startswith("Title. ")
    assert "Sentence number 2" in chunks[0] and "Sentence number 2" in chunks[1]  # the overlap
    assert all(len(c) <= 500 for c in chunks)
    assert reviews_index.chunk_review("", "x" * 600, 1000, 3, 2, 100, False) == ["x" * 100]


def test_build_chunks_keeps_review_metadata(fixtures_dir: Path) -> None:
    rows = [json.loads(x) for x in (fixtures_dir / "reviews_sample.jsonl").read_text().splitlines()]
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms")
    chunks = reviews_index.build_chunks(frame)
    assert set(chunks["item_id"]) == {"B0AAA", "B0BBB", "B0CCC"}
    assert chunks["review_id"].str.startswith("R").all() and chunks["chunk_id"].is_unique
    first = chunks.iloc[0]
    assert first["rating"] == 5.0 and first["date"] == "2020-09-13"
    stats = reviews_index.chunk_stats(chunks)
    assert stats["reviews"] == len(frame) and stats["chunks"] == len(chunks)


def test_split_sentences() -> None:
    assert reviews_index.split_sentences("It fits. Really well! Love it?") == [
        "It fits.",
        "Really well!",
        "Love it?",
    ]


# ------------------------------------------------------------------------------- vision helpers
def test_label_maps_use_the_train_split_only_and_unseen_labels_are_ignored() -> None:
    index = pd.DataFrame(
        {
            "split": ["train", "train", "val", "test"],
            "articleType": ["Shirts", "Tops", "Watches", None],
        }
    )
    maps = vdata.build_label_maps(index, ["articleType"])
    assert maps == {"articleType": ["Shirts", "Tops"]}  # "Watches" only appears in validation
    enc = vdata.encode_labels(index, maps)["articleType"].tolist()
    assert enc == [0, 1, vdata.IGNORE_INDEX, vdata.IGNORE_INDEX]


def test_save_label_maps_and_stratified_subset(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "maps.json"
    vdata.save_label_maps({"a": ["x", "y"]}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": ["x", "y"]}
    index = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 2})
    pos = vdata.stratified_subset(index, 0.3, seed=1, col="c")
    assert set(index.loc[pos, "c"]) == {"a", "b"} and len(pos) == 3 + 1  # at least one per class
    assert len(vdata.stratified_subset(index, 1.0, seed=1, col="c")) == 12


def test_tensor_helpers_shapes_and_ranges() -> None:
    batch = np.full((2, 20, 10, 3), 255, dtype=np.uint8)
    x = vdata.to_tensor(batch)
    assert x.shape == (2, 3, 20, 10) and float(x.max()) == 1.0
    assert vdata.resize(x, (40, 30)).shape == (2, 3, 40, 30)
    z = vdata.normalise(x)
    assert z.shape == x.shape and float(z[0, 0, 0, 0]) == pytest.approx((1 - 0.485) / 0.229)


def test_augment_rejects_hue_shift() -> None:
    cfg = {"hue": 0.1}
    with pytest.raises(ValueError, match="colour labels"):
        vdata.augment(torch.rand(1, 3, 8, 8), cfg, torch.Generator().manual_seed(0))


def test_evaluation_tables() -> None:
    classes = ["a", "b", "c"]
    y = np.array([0, 0, 1, 1, 2, 2])
    probs = np.eye(3)[[0, 0, 1, 2, 2, 2]] * 0.9 + 0.033
    table = per_class_table(y, probs, classes)
    assert set(table["class"]) == set(classes) and table["support"].sum() == 6
    cm, names = confusion_top_k(y, probs, classes, k=2)
    assert names[-1] == "(other)" and cm.shape == (2, 3) and cm.sum() == 4
    bins = reliability_bins(y, probs, n_bins=5)
    assert bins["count"].sum() == 6 and (bins["accuracy"].between(0, 1)).all()


def test_apply_threshold_reports_coverage_and_precision() -> None:
    y = np.array([0, 1, 1, vdata.IGNORE_INDEX])
    probs = np.array([[0.9, 0.1], [0.6, 0.4], [0.2, 0.8], [0.5, 0.5]])
    out = apply_threshold(y, probs, 0.7)
    assert out["coverage"] == pytest.approx(2 / 3) and out["precision"] == pytest.approx(1.0)
    none = apply_threshold(y, probs, 0.99)
    assert none["coverage"] == 0.0 and np.isnan(none["precision"])


def test_multihead_net_freezing_and_batchnorm_mode() -> None:
    net = MultiHeadNet({"a": 3, "b": 2}, pretrained=False)
    net.set_backbone_trainable(None)
    assert not any(p.requires_grad for p in net.features.parameters())
    net.set_backbone_trainable(len(net.features) - 1)
    assert any(p.requires_grad for p in net.features[-1].parameters())
    net.train()
    frozen_bn = [
        m
        for m in net.features[0].modules()
        if isinstance(m, torch.nn.BatchNorm2d) and not m.training
    ]
    assert frozen_bn  # frozen blocks keep their BatchNorm statistics
    out = net(torch.rand(2, 3, 80, 60))
    assert out["a"].shape == (2, 3) and out["b"].shape == (2, 2)
    assert net.embed(torch.rand(1, 3, 80, 60)).shape[0] == 1


def test_verified_mask_requires_agreement_and_confidence() -> None:
    catalog = pd.DataFrame({"id": [1, 2, 3], "articleType": ["Shirts", "Tops", "Watches"]})
    check = pd.DataFrame(
        {
            "id": [1, 2],
            "pred_articleType": ["Shirts", "Shirts"],
            "pred_confidence": [0.95, 0.99],
        }
    )
    assert verified_mask(catalog, check, 0.9).tolist() == [True, False, False]
    assert verified_mask(catalog, check, 0.99).tolist() == [False, False, False]


def test_most_confident_wrong_lists_the_surest_mistakes_first() -> None:
    from catalogiq.vision.errors import most_confident_wrong

    classes = ["a", "b"]
    y = np.array([0, 0, 1, vdata.IGNORE_INDEX, 1])
    probs = np.array([[0.1, 0.9], [0.4, 0.6], [0.2, 0.8], [0.0, 1.0], [0.3, 0.7]])
    out = most_confident_wrong(y, probs, classes, ids=np.array([10, 11, 12, 13, 14]), k=5)
    assert out["id"].tolist() == [10, 11]  # rows 12 and 14 are right; row 13 has no label
    assert out["true"].tolist() == ["a", "a"] and out["predicted"].tolist() == ["b", "b"]
    assert out["confidence"].is_monotonic_decreasing
