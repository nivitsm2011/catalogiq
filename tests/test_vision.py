"""Unit tests for the attribute classifier (tiny random models, no pretrained downloads)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from catalogiq.config import get_vision_config
from catalogiq.vision.calibration import (
    apply_temperature,
    choose_threshold,
    fit_temperature,
)
from catalogiq.vision.data import IGNORE_INDEX, augment, class_weights
from catalogiq.vision.evaluate import expected_calibration_error, score_target
from catalogiq.vision.model import ExportWrapper, MultiHeadNet
from catalogiq.vision.predict import AttributePredictor, measure_latency


def test_augment_shape_range_and_no_hue_shift() -> None:
    cfg = get_vision_config()["augmentation"]
    assert cfg["hue"] == 0.0
    x = torch.rand(8, 3, 32, 24)
    out = augment(x, cfg, torch.Generator().manual_seed(0))
    assert out.shape == x.shape and 0.0 <= float(out.min()) and float(out.max()) <= 1.0
    with pytest.raises(ValueError):
        augment(x, {**cfg, "hue": 0.1}, torch.Generator().manual_seed(0))


def test_augment_keeps_pure_blue_blue() -> None:
    """A pure blue image stays blue-dominant after augmentation (colour labels stay valid)."""
    cfg = get_vision_config()["augmentation"]
    x = torch.zeros(16, 3, 32, 24)
    x[:, 2] = 0.8
    out = augment(x, cfg, torch.Generator().manual_seed(1))
    assert bool((out[:, 2].mean(dim=(1, 2)) > out[:, 0].mean(dim=(1, 2))).all())
    assert bool((out[:, 2].mean(dim=(1, 2)) > out[:, 1].mean(dim=(1, 2))).all())


def test_class_weights_favour_rare_classes() -> None:
    labels = torch.tensor([0] * 90 + [1] * 9 + [2] * 1 + [IGNORE_INDEX] * 5)
    w = class_weights(labels, 3, power=0.5)
    assert w[2] > w[1] > w[0]
    assert float(w.mean()) == pytest.approx(1.0, rel=1e-5)


def test_temperature_scaling_reduces_overconfidence() -> None:
    gen = torch.Generator().manual_seed(0)
    labels = torch.randint(0, 4, (2000,), generator=gen)
    logits = torch.randn(2000, 4, generator=gen) * 5.0  # wildly overconfident, random
    logits[torch.arange(2000), labels] += 4.0
    before = expected_calibration_error(labels.numpy(), torch.softmax(logits, -1).numpy())
    t = fit_temperature(logits, labels)
    after = expected_calibration_error(labels.numpy(), apply_temperature(logits, t))
    assert t > 1.0 and after < before


def test_choose_threshold_reaches_target_precision() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.3, 1.0, 5000)
    correct = rng.uniform(size=5000) < conf  # calibrated by construction
    probs = np.stack([conf, 1 - conf], axis=1)
    y = np.where(correct, 0, 1)
    out = choose_threshold(y, probs, target_precision=0.9)
    assert out["precision"] >= 0.9 and 0.0 < out["coverage"] < 1.0
    unreachable = choose_threshold(y, probs, target_precision=1.01)
    assert unreachable["coverage"] == 0.0


def test_score_target_ignores_missing_labels() -> None:
    probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    y = np.array([0, 1, IGNORE_INDEX])
    assert score_target(y, probs)["accuracy"] == 1.0


@pytest.fixture()
def tiny_model_dir(tmp_path):
    """Export a randomly initialised model in the same layout the real exporter writes."""
    v = get_vision_config()
    labels = {"articleType": ["Shirts", "Tshirts", "Watches"], "gender": ["Men", "Women"]}
    net = MultiHeadNet({t: len(c) for t, c in labels.items()}, pretrained=False).eval()
    h, w = v["input_hw"]
    scripted = torch.jit.trace(ExportWrapper(net), torch.randn(1, 3, h, w))
    scripted.save(str(tmp_path / v["inference"]["torchscript_file"]))
    (tmp_path / v["inference"]["labels_file"]).write_text(json.dumps(labels), encoding="utf-8")
    thresholds = {t: {"temperature": 1.0, "threshold": 0.99} for t in labels}
    (tmp_path / v["inference"]["thresholds_file"]).write_text(json.dumps(thresholds), "utf-8")
    return tmp_path


def test_predict_returns_label_confidence_review_flag(tiny_model_dir) -> None:
    predictor = AttributePredictor(tiny_model_dir)
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (80, 60, 3), dtype=np.uint8))
    result = predictor.predict(img)
    assert set(result) == {"articleType", "gender"}
    for label, conf, needs_review in result.values():
        assert isinstance(label, str) and 0.0 <= conf <= 1.0 and isinstance(needs_review, bool)
    assert result["articleType"][0] in {"Shirts", "Tshirts", "Watches"}
    # a random network is never 99% sure on 2-3 classes, so everything must be flagged
    assert all(r[2] for r in result.values())


def test_predict_accepts_array_path_and_other_sizes(tiny_model_dir, tmp_path) -> None:
    predictor = AttributePredictor(tiny_model_dir)
    arr = np.zeros((200, 150, 3), dtype=np.uint8)
    path = tmp_path / "x.jpg"
    Image.fromarray(arr).save(path)
    assert predictor.predict(arr).keys() == predictor.predict(path).keys()
    assert predictor.predict(Image.fromarray(arr).convert("L")).keys() == {"articleType", "gender"}


def test_measure_latency_reports_positive_times(tiny_model_dir) -> None:
    predictor = AttributePredictor(tiny_model_dir)
    img = Image.fromarray(np.zeros((80, 60, 3), dtype=np.uint8))
    stats = measure_latency(predictor, img, runs=5)
    assert stats["mean_ms"] > 0 and stats["p95_ms"] >= stats["median_ms"] > 0
