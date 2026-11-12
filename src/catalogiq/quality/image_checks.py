"""Image quality checks with OpenCV: blur, exposure, resolution, aspect ratio, background, text.

Every measure is a plain number computed from the pixels; a flag is raised when a number crosses a
threshold. Thresholds come from ``configs/quality.yaml`` and are overridden by the values tuned in
``scripts/tune_quality_thresholds.py`` (``reports/phase6/thresholds.json``).
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from catalogiq.config import get_quality_config, resolve_path

CHECKS = (
    "blurry",
    "too_dark",
    "overexposed",
    "low_resolution",
    "bad_aspect_ratio",
    "cluttered_background",
    "text_or_watermark",
)


def load_thresholds() -> dict[str, float]:
    """Defaults from the config, replaced by tuned values when ``thresholds.json`` exists."""
    cfg = get_quality_config()
    thresholds = dict(cfg["image"]["thresholds"])
    path = resolve_path(cfg["paths"]["thresholds"])
    if path.exists():
        tuned = json.loads(path.read_text(encoding="utf-8"))
        thresholds.update({k: v for k, v in tuned.items() if k in thresholds})
    return thresholds


def _work_gray(img: np.ndarray, long_side: int) -> tuple[np.ndarray, np.ndarray]:
    """RGB uint8 -> (RGB, gray), shrunk so the long side is at most ``long_side``."""
    h, w = img.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), cv2.INTER_AREA)
    return img, cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def corner_std(rgb: np.ndarray, frac: float) -> float:
    """Median colour spread over the four corner patches.

    A product sits in the middle, so on a clean photo the corners are plain and the spread is
    near 0, while a busy background is busy in all four corners. (A border band was tried first
    and failed: models cross it, so clean photos looked cluttered.)
    """
    h, w = rgb.shape[:2]
    ch, cw = max(2, int(h * frac)), max(2, int(w * frac))
    patches = [rgb[:ch, :cw], rgb[:ch, w - cw :], rgb[h - ch :, :cw], rgb[h - ch :, w - cw :]]
    return float(np.median([p.astype(np.float32).std(axis=(0, 1)).mean() for p in patches]))


def text_row_score(gray: np.ndarray) -> int:
    """Largest number of text-like blobs that share a text line (0 when there is none).

    Heuristic: edge-strengthen the picture, binarise, and keep small connected blobs of letter size;
    blobs whose centres lie on one horizontal line and have similar heights count as characters.
    Patterns such as stripes or lace can trigger it (see the evaluation).
    """
    h, w = gray.shape
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((2, 2), np.uint8))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    n, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 3 <= bh <= 0.3 * h and 2 <= bw <= 0.3 * w and area >= 4 and 0.2 <= bw / bh <= 1.8:
            blobs.append((cents[i][1], bh))
    best = 0
    for cy, bh in blobs:
        row = [b for c, b in blobs if abs(c - cy) <= max(2.0, 0.35 * bh) and 0.6 <= b / bh <= 1.7]
        best = max(best, len(row))
    return best


def compute_metrics(img: np.ndarray) -> dict[str, float]:
    """Numbers behind every check for one RGB uint8 image.

    Returns:
        ``blur`` (variance of the Laplacian), ``mean_luma``, ``p5_luma``, ``short_side``, ``aspect``
        (width / height), ``corner_std`` (background busyness) and ``text_chars`` (blob heuristic,
        kept as the baseline for the CLIP text check).
    """
    cfg = get_quality_config()["image"]
    h, w = img.shape[:2]
    rgb, gray = _work_gray(img, cfg["work_long_side"])
    return {
        "blur": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "mean_luma": float(gray.mean()),
        "p5_luma": float(np.percentile(gray, 5)),
        "short_side": float(min(h, w)),
        "aspect": float(w / h),
        "corner_std": corner_std(rgb, cfg["corner_frac"]),
        "text_chars": float(text_row_score(gray)),
    }


def flags_from_metrics(
    m: dict[str, float], t: dict[str, float], text_margin: float | None = None
) -> dict[str, bool]:
    """Apply the thresholds to the measured numbers (one boolean per check).

    ``text_margin`` is the CLIP text-overlay score from ``clip_checks``; without it the text check
    is skipped (False).
    """
    return {
        "blurry": m["blur"] < t["blur_min_laplacian_var"],
        "too_dark": m["mean_luma"] < t["dark_max_mean_luma"],
        "overexposed": m["p5_luma"] > t["bright_min_p5_luma"],
        "low_resolution": m["short_side"] < t["min_short_side_px"],
        "bad_aspect_ratio": not (t["aspect_min"] <= m["aspect"] <= t["aspect_max"]),
        "cluttered_background": m["corner_std"] > t["clutter_max_corner_std"],
        "text_or_watermark": text_margin is not None and text_margin > t["text_min_clip_margin"],
    }


def check_image(
    img: np.ndarray,
    thresholds: dict[str, float] | None = None,
    text_margin: float | None = None,
) -> dict[str, Any]:
    """Run every image check on one RGB uint8 image.

    Returns:
        ``{"metrics": {...}, "flags": {check: bool}, "issues": [names of raised checks]}``.
    """
    metrics = compute_metrics(img)
    flags = flags_from_metrics(metrics, thresholds or load_thresholds(), text_margin)
    return {"metrics": metrics, "flags": flags, "issues": [k for k, v in flags.items() if v]}
