"""Image grids of model mistakes for manual review (error analysis).

Usage: python scripts/error_grids.py
Outputs: reports/figures/06_confident_errors.png, reports/figures/07_random_errors.png,
         reports/error_samples.csv
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.vision.data import IGNORE_INDEX, build_image_cache, encode_labels
from catalogiq.vision.errors import most_confident_wrong


def draw(df: pd.DataFrame, path: str, cols: int, title: str, images_dir) -> None:
    """Draw one tile per error with 'true -> predicted (confidence)'."""
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, 2.9 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).ravel(), df.iterrows(), strict=False):
        ax.imshow(Image.open(image_path(images_dir, int(r["id"]))))
        ax.set_title(
            f"#{int(r['n'])} id {int(r['id'])}\ntrue: {r['true']}\npred: {r['predicted']}\n"
            f"conf {r['confidence']:.2f}",
            fontsize=7,
            loc="left",
        )
    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(resolve_path(path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Build both grids for articleType errors on the test split."""
    cfg, v = get_config(), get_vision_config()
    _, index = build_image_cache()
    labels_json = json.loads(
        (resolve_path(v["inference"]["model_dir"]) / v["inference"]["labels_file"]).read_text()
    )
    label_maps = {t: labels_json[t] for t in v["targets"]}
    labels = encode_labels(index, label_maps)
    test_pos = np.where(index["split"].to_numpy() == "test")[0]
    ids = index["id"].to_numpy()[test_pos]
    probs = np.load(resolve_path("data/interim/test_probs_full.npz"))["articleType"]
    y = labels["articleType"][test_pos].numpy()
    images_dir = resolve_path(cfg["data"]["fashion"]["images_dir"])

    top = most_confident_wrong(y, probs, label_maps["articleType"], ids, k=24)
    top.insert(0, "n", range(1, len(top) + 1))
    draw(
        top,
        "reports/figures/06_confident_errors.png",
        6,
        "24 most confident wrong articleType predictions (test split)",
        images_dir,
    )

    wrong = np.where((y != IGNORE_INDEX) & (probs.argmax(1) != y))[0]
    rng = np.random.default_rng(cfg["project"]["seed"])
    pick = np.sort(rng.choice(wrong, size=40, replace=False))
    sample = pd.DataFrame(
        {
            "n": range(1, len(pick) + 1),
            "id": ids[pick],
            "true": [label_maps["articleType"][y[i]] for i in pick],
            "predicted": [label_maps["articleType"][probs[i].argmax()] for i in pick],
            "confidence": probs[pick].max(1).round(4),
        }
    )
    draw(
        sample,
        "reports/figures/07_random_errors.png",
        8,
        "40 randomly chosen wrong articleType predictions (test split)",
        images_dir,
    )
    top.to_csv(resolve_path("reports/error_top24.csv"), index=False)
    sample.to_csv(resolve_path("reports/error_samples.csv"), index=False)
    print(f"articleType test errors: {len(wrong)} of {int((y != IGNORE_INDEX).sum())}")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
