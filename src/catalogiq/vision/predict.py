"""Inference: photo -> {attribute: (label, confidence, needs_review)}.

Loads the exported TorchScript model, the class lists and the per-target temperature and confidence
threshold produced by ``scripts/evaluate_classifier.py``. Preprocessing is identical to training
(resize to the cached size, then to the network input size, then ImageNet normalisation).
"""

from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from catalogiq.config import get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.vision.data import normalise, resize, to_tensor

logger = get_logger(__name__)

ImageLike = Image.Image | np.ndarray | str | Path
Prediction = dict[str, tuple[str, float, bool]]


class AttributePredictor:
    """TorchScript attribute classifier with calibrated confidences and review flags.

    Args:
        model_dir: Folder with the TorchScript file, ``labels.json`` and ``thresholds.json``.
            Defaults to ``vision.inference.model_dir`` from the config.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        cfg = get_vision_config()
        icfg = cfg["inference"]
        self.model_dir = resolve_path(model_dir or icfg["model_dir"])
        self.stored_hw = tuple(cfg["stored_hw"])
        self.input_hw = tuple(cfg["input_hw"])
        self.model = torch.jit.load(str(self.model_dir / icfg["torchscript_file"]))
        self.model.eval()
        labels = json.loads((self.model_dir / icfg["labels_file"]).read_text(encoding="utf-8"))
        self.targets: list[str] = list(labels)
        self.labels: dict[str, list[str]] = labels
        thresholds = json.loads(
            (self.model_dir / icfg["thresholds_file"]).read_text(encoding="utf-8")
        )
        self.temperature = {t: thresholds[t]["temperature"] for t in self.targets}
        self.threshold = {t: thresholds[t]["threshold"] for t in self.targets}

    @staticmethod
    def _to_pil(image: ImageLike) -> Image.Image:
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        return image.convert("RGB")

    def preprocess(self, image: ImageLike) -> torch.Tensor:
        """Image -> normalised (1, 3, H, W) tensor."""
        h, w = self.stored_hw
        small = np.asarray(self._to_pil(image).resize((w, h), Image.BILINEAR))
        return normalise(resize(to_tensor(small[None]), self.input_hw))

    @torch.no_grad()
    def predict(self, image: ImageLike) -> Prediction:
        """Predict every attribute for one image.

        Returns:
            ``{attribute: (label, confidence, needs_review)}``. ``needs_review`` is True when the
            calibrated confidence is below that attribute's threshold ("not sure - please confirm").
        """
        logits = self.model(self.preprocess(image))
        result: Prediction = {}
        for target, lg in zip(self.targets, logits, strict=True):
            probs = torch.softmax(lg[0] / self.temperature[target], dim=-1)
            conf, idx = probs.max(dim=0)
            result[target] = (
                self.labels[target][int(idx)],
                float(conf),
                bool(float(conf) < self.threshold[target]),
            )
        return result


@lru_cache(maxsize=2)
def _default_predictor(model_dir: str | None) -> AttributePredictor:
    return AttributePredictor(model_dir)


def predict(image: ImageLike, model_dir: str | Path | None = None) -> Prediction:
    """Predict attributes for one image using a cached predictor.

    Args:
        image: PIL image, RGB/gray ``numpy`` array or a file path.
        model_dir: Optional override of the exported model folder.

    Returns:
        ``{attribute: (label, confidence, needs_review)}``.
    """
    return _default_predictor(str(model_dir) if model_dir else None).predict(image)


def measure_latency(
    predictor: AttributePredictor, image: ImageLike, runs: int | None = None
) -> dict[str, Any]:
    """Wall-clock CPU latency of ``predict`` (preprocessing included) in milliseconds."""
    runs = runs or get_vision_config()["inference"]["latency_runs"]
    for _ in range(5):  # warm-up
        predictor.predict(image)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        predictor.predict(image)
        times.append((time.perf_counter() - start) * 1000)
    arr = np.array(times)
    return {
        "runs": runs,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "threads": torch.get_num_threads(),
    }
