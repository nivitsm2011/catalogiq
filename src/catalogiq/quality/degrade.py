"""Make degraded copies of good photos so each image check can be scored against known defects.

Every function takes and returns an RGB uint8 array. ``make_degraded_set`` yields one copy per
(defect type, level) with the type and level recorded, so recall can be reported by severity.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np

from catalogiq.config import get_quality_config

WORDS = ["SAMPLE", "SHOP123", "50% OFF", "www.shop.com", "(c) PHOTO", "SALE", "NEW"]


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur (out-of-focus or shaky photo)."""
    return cv2.GaussianBlur(img, (0, 0), sigma)


def darken(img: np.ndarray, factor: float) -> np.ndarray:
    """Multiply brightness by ``factor`` < 1 (photo taken in dim light)."""
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def wash_out(img: np.ndarray, blend: float) -> np.ndarray:
    """Blend with white (overexposed, flash glare)."""
    return np.clip(img.astype(np.float32) * (1 - blend) + 255 * blend, 0, 255).astype(np.uint8)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    """Shrink the photo (a small upload); the smaller size is kept."""
    h, w = img.shape[:2]
    return cv2.resize(img, (max(2, round(w * factor)), max(2, round(h * factor))), cv2.INTER_AREA)


def stretch(img: np.ndarray, aspect: float) -> np.ndarray:
    """Resize to a new width / height ratio at the same height (squeezed or stretched photo)."""
    h = img.shape[0]
    return cv2.resize(img, (max(2, round(h * aspect)), h), cv2.INTER_LINEAR)


def _foreground(img: np.ndarray) -> np.ndarray:
    """Pixels that differ from the typical border colour (the product on a plain background)."""
    h, w = img.shape[:2]
    band = np.ones((h, w), bool)
    band[3 : h - 3, 3 : w - 3] = False
    bg = np.median(img[band], axis=0)
    mask = (np.abs(img.astype(np.float32) - bg).max(axis=2) > 30).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return mask.astype(bool)


def clutter(
    img: np.ndarray, rng: np.random.Generator, other: np.ndarray | None = None
) -> np.ndarray:
    """Paste the product onto a busy background (coloured blocks or another photo)."""
    h, w = img.shape[:2]
    if other is not None and rng.random() < 0.5:
        bg = cv2.resize(other, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        bg = np.empty_like(img)
        bg[:] = rng.integers(30, 200, 3)  # mid-tone base so the background is never near-white
        for _ in range(45):  # many small patches: a busy room, shelf or pattern
            x0, y0 = int(rng.integers(0, w - 3)), int(rng.integers(0, h - 3))
            x1, y1 = x0 + int(rng.integers(4, w // 3)), y0 + int(rng.integers(4, h // 4))
            bg[y0:y1, x0:x1] = rng.integers(0, 256, 3)
        bg = np.clip(bg.astype(np.int16) + rng.integers(-18, 19, bg.shape), 0, 255).astype(np.uint8)
    out = bg.copy()
    mask = _foreground(img)
    out[mask] = img[mask]
    return out


def overlay_text(img: np.ndarray, rng: np.random.Generator, style: str | None = None) -> np.ndarray:
    """Overlay a text banner (solid) or a semi-transparent diagonal watermark."""
    h, w = img.shape[:2]
    style = style or ("banner" if rng.random() < 0.5 else "watermark")
    word = WORDS[int(rng.integers(len(WORDS)))]
    scale = float(rng.uniform(0.32, 0.5))
    layer = np.zeros_like(img)
    colour = tuple(int(c) for c in rng.choice([0, 128, 255], 3))
    (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(rng.integers(0, max(1, w - tw)))
    y = int(rng.integers(th + 2, max(th + 3, h - 2)))
    cv2.putText(layer, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour or (255, 255, 255), 1)
    drawn = layer.sum(axis=2) > 0
    if style == "watermark":
        mat = cv2.getRotationMatrix2D((w / 2, h / 2), float(rng.uniform(-35, 35)), 1.0)
        drawn = cv2.warpAffine(drawn.astype(np.uint8), mat, (w, h)) > 0
        layer = cv2.warpAffine(layer, mat, (w, h))
        alpha = float(rng.uniform(0.45, 0.8))
        out = img.copy().astype(np.float32)
        out[drawn] = out[drawn] * (1 - alpha) + layer[drawn] * alpha
        return out.astype(np.uint8)
    out = img.copy()
    out[drawn] = layer[drawn]
    return out


def make_degraded_set(
    images: np.ndarray, rng: np.random.Generator
) -> Iterator[tuple[int, str, int, np.ndarray]]:
    """Yield ``(source_index, defect_type, level, degraded_image)`` for every degradation.

    Types: blur, dark, bright, lowres, aspect (one entry per level in ``quality.degrade``), plus
    clutter and text (one random version each, level 0).
    """
    d = get_quality_config()["degrade"]
    for i, img in enumerate(images):
        for lv, v in enumerate(d["blur_sigma"]):
            yield i, "blur", lv, blur(img, v)
        for lv, v in enumerate(d["darken_factor"]):
            yield i, "dark", lv, darken(img, v)
        for lv, v in enumerate(d["wash_blend"]):
            yield i, "bright", lv, wash_out(img, v)
        for lv, v in enumerate(d["downscale_factor"]):
            yield i, "lowres", lv, downscale(img, v)
        for lv, v in enumerate(d["stretch_aspect"]):
            yield i, "aspect", lv, stretch(img, v)
        other = images[int(rng.integers(len(images)))]
        yield i, "clutter", 0, clutter(img, rng, other)
        yield i, "text", 0, overlay_text(img, rng)
