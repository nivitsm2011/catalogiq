"""Evaluate the image checks on the TEST split with the thresholds tuned on validation.

500 clean test photos plus 17 degraded copies of each (blur, dark, bright, low-res, stretched at
three levels; one cluttered; one with overlaid text). Writes reports/phase6/image_eval_flags.csv.
Usage: python scripts/evaluate_image_checks.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.quality.clip_checks import prompt_embeddings, text_margin
from catalogiq.quality.degrade import make_degraded_set
from catalogiq.quality.image_checks import compute_metrics, flags_from_metrics, load_thresholds
from catalogiq.search.embeddings import ClipEncoder

logger = get_logger("evaluate_image_checks")
N_CLEAN = 500


def main() -> None:
    """Score every clean and degraded test photo with every check."""
    cfg = get_quality_config()
    rng = np.random.default_rng(cfg["seed"] + 1)
    images = np.load(resolve_path(cfg["paths"]["image_cache"]), mmap_mode="r")
    index = pd.read_parquet(resolve_path(cfg["paths"]["image_index"]))
    test = np.where(index["split"].eq("test"))[0]
    pick = rng.choice(test, N_CLEAN, replace=False)
    clean = np.asarray(images[pick])
    thresholds = load_thresholds()
    encoder = ClipEncoder()
    prompts = prompt_embeddings(encoder)
    items: list[tuple[str, str, int, np.ndarray]] = [("clean", "", -1, a) for a in clean]
    items += [("degraded", t, lv, out) for _, t, lv, out in make_degraded_set(clean, rng)]
    logger.info("%d photos to score", len(items))
    rows = []
    for k in range(0, len(items), 256):
        chunk = items[k : k + 256]
        # CLIP wants a fixed-size batch; resize every image to 80x60 only for the CLIP text check
        batch = np.stack([_to_80x60(c[3]) for c in chunk])
        margins = text_margin(encoder.encode_images_u8(batch), prompts)
        for (kind, dtype, lv, img), m in zip(chunk, margins, strict=True):
            metrics = compute_metrics(img)
            rows.append(
                {
                    "kind": kind,
                    "type": dtype,
                    "level": lv,
                    **metrics,
                    "text_margin": float(m),
                    **flags_from_metrics(metrics, thresholds, float(m)),
                }
            )
        logger.info("scored %d / %d", min(k + 256, len(items)), len(items))
    out = resolve_path(cfg["paths"]["eval_dir"]) / "image_eval_flags.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info("wrote %s", out)


def _to_80x60(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.resize(img, (60, 80), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
