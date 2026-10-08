"""Grad-CAM figure: what the model looks at for correct and for confidently wrong predictions.

Usage: python scripts/gradcam_figure.py
Output: reports/figures/08_gradcam.png
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.utils import configure_cache_dirs
from catalogiq.vision.data import build_image_cache, normalise, resize, to_tensor
from catalogiq.vision.gradcam import grad_cam
from catalogiq.vision.model import MultiHeadNet

CORRECT_TYPES = ["Tshirts", "Watches", "Casual Shoes", "Handbags"]
WRONG_IDS = [24459, 13796, 44760, 56207]  # briefs-as-shoes, sweater-as-tshirt, sneaker, pendant


def main() -> None:
    """Render a 2x4 grid: top row correct predictions, bottom row confident mistakes."""
    configure_cache_dirs()
    cfg, v = get_config(), get_vision_config()
    model_dir = resolve_path(v["inference"]["model_dir"])
    labels = json.loads((model_dir / v["inference"]["labels_file"]).read_text())
    net = MultiHeadNet({t: len(c) for t, c in labels.items()}, pretrained=False)
    net.load_state_dict(torch.load(model_dir / "best_full.pt"))
    net.eval()
    images, index = build_image_cache()
    classes = labels["articleType"]
    probs = np.load(resolve_path("data/interim/test_probs_full.npz"))["articleType"]
    test = index[index["split"] == "test"].reset_index(drop=True)
    pred = probs.argmax(1)
    conf = probs.max(1)
    pos_by_id = {int(i): p for p, i in enumerate(test["id"])}
    chosen: list[tuple[int, str]] = []
    for t in CORRECT_TYPES:
        ok = np.where((test["articleType"] == t) & (np.array(classes)[pred] == t))[0]
        chosen.append((int(ok[np.argmax(conf[ok])]), "correct"))
    chosen += [(pos_by_id[i], "wrong") for i in WRONG_IDS]

    full_pos = {int(i): p for p, i in enumerate(index["id"])}
    fig, axes = plt.subplots(2, 4, figsize=(12, 7.6))
    hw = tuple(v["input_hw"])
    images_dir = resolve_path(cfg["data"]["fashion"]["images_dir"])
    for ax, (tpos, kind) in zip(axes.ravel(), chosen, strict=True):
        pid = int(test["id"][tpos])
        x = normalise(resize(to_tensor(images[full_pos[pid]][None]), hw))
        cam, idx = grad_cam(net, x, "articleType")
        photo = np.asarray(Image.open(image_path(images_dir, pid)).convert("RGB").resize(hw[::-1]))
        ax.imshow(photo)
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_title(
            f"{kind.upper()} | true: {test['articleType'][tpos]}\n"
            f"pred: {classes[idx]} ({conf[tpos]:.2f})",
            fontsize=8,
            color="#1b7f3b" if kind == "correct" else "#b3261e",
        )
        ax.axis("off")
    fig.suptitle("Grad-CAM (articleType head): red = regions that raised the predicted class")
    plt.tight_layout()
    plt.savefig(resolve_path("reports/figures/08_gradcam.png"), dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    main()
