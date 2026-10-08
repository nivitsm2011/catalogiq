"""Export the trained classifier to TorchScript and measure CPU inference latency.

Usage: python scripts/export_model.py --tag full
Outputs: models/attribute_classifier/model_ts.pt, reports/latency.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from PIL import Image

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.utils import configure_cache_dirs
from catalogiq.vision.data import build_image_cache
from catalogiq.vision.model import ExportWrapper, MultiHeadNet
from catalogiq.vision.predict import AttributePredictor, measure_latency

logger = get_logger(__name__)


def main() -> None:
    """Trace the model, check it matches the eager model, then time the predictor."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    configure_cache_dirs()
    v = get_vision_config()
    model_dir = resolve_path(v["inference"]["model_dir"])
    labels = json.loads((model_dir / v["inference"]["labels_file"]).read_text(encoding="utf-8"))
    net = MultiHeadNet(
        {t: len(c) for t, c in labels.items()},
        feature_dim=v["backbone"]["feature_dim"],
        pretrained=False,
    )
    net.load_state_dict(torch.load(model_dir / f"best_{args.tag}.pt"))
    net.eval()
    h, w = v["input_hw"]
    example = torch.randn(1, 3, h, w)
    with torch.no_grad():
        scripted = torch.jit.trace(ExportWrapper(net), example)
        batch = torch.randn(5, 3, h, w)
        eager = [net(batch)[t] for t in net.targets]
        traced = scripted(batch)
    max_diff = max(float((a - b).abs().max()) for a, b in zip(eager, traced, strict=True))
    logger.info("max |eager - torchscript| on a batch of 5: %.2e", max_diff)
    assert max_diff < 1e-4, "TorchScript output differs from the eager model"
    out_path = model_dir / v["inference"]["torchscript_file"]
    scripted.save(str(out_path))
    logger.info("saved %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    images, index = build_image_cache()
    sample = Image.fromarray(np.asarray(images[0]))
    predictor = AttributePredictor()
    latency = measure_latency(predictor, sample)
    latency["torchscript_mb"] = round(out_path.stat().st_size / 1e6, 2)
    latency["note"] = "single image, CPU, preprocessing included, x64 Python emulated on ARM64"
    resolve_path("reports/latency.json").write_text(json.dumps(latency, indent=2), encoding="utf-8")
    logger.info("latency: %s", latency)
    _ = get_config()


if __name__ == "__main__":
    main()
