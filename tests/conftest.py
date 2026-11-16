"""Shared test helpers: paths to the small committed fixtures in tests/fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from catalogiq.config import get_vision_config
from catalogiq.vision.model import ExportWrapper, MultiHeadNet

FIXTURES = Path(__file__).resolve().parent / "fixtures"
PHOTOS = FIXTURES / "photos"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Folder with the committed photos, a styles sample and review/meta JSONL samples."""
    return FIXTURES


@pytest.fixture(scope="session")
def photo_dir() -> Path:
    """Folder with the small test photos (clean, blurry, dark, cluttered, tiny, other formats)."""
    return PHOTOS


def load_rgb(name: str) -> np.ndarray:
    """A fixture photo as an RGB uint8 array."""
    with Image.open(PHOTOS / name) as img:
        return np.asarray(img.convert("RGB"))


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
