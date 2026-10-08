"""Data layer for the attribute classifier.

Images are decoded once into a uint8 array (``image_cache``) so that training never touches JPEG
decoding again, and augmentation is applied to whole batches with plain torch ops (fast on CPU, no
DataLoader worker processes needed on Windows).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_INDEX = -100


def build_image_cache(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Decode every cleaned image once, resize to ``stored_hw`` and cache as a uint8 array.

    Returns:
        Array of shape (N, H, W, 3) and the matching index frame (``id``, ``split``, targets...).
    """
    cfg, vcfg = get_config(), get_vision_config()
    cache_path, index_path = resolve_path(vcfg["image_cache"]), resolve_path(vcfg["image_index"])
    if cache_path.exists() and index_path.exists() and not force:
        return np.load(cache_path, mmap_mode="r"), pd.read_parquet(index_path)
    clean = pd.read_parquet(resolve_path(cfg["data"]["fashion"]["clean_parquet"]))
    h, w = vcfg["stored_hw"]
    images_dir = resolve_path(cfg["data"]["fashion"]["images_dir"])
    array = np.empty((len(clean), h, w, 3), dtype=np.uint8)
    for i, pid in enumerate(tqdm(clean["id"].to_numpy(), desc="caching images")):
        with Image.open(image_path(images_dir, pid)) as img:
            array[i] = np.asarray(img.convert("RGB").resize((w, h), Image.BILINEAR))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, array)
    keep = ["id", "split", "dup_group", *vcfg["targets"]]
    clean[keep].to_parquet(index_path, index=False)
    logger.info("cached %d images -> %s (%.0f MB)", len(clean), cache_path, array.nbytes / 1e6)
    return array, clean[keep].reset_index(drop=True)


def build_label_maps(index: pd.DataFrame, targets: list[str]) -> dict[str, list[str]]:
    """Class lists per target, fitted on the TRAIN split only (sorted for determinism)."""
    train = index[index["split"] == "train"]
    return {t: sorted(train[t].dropna().unique().tolist()) for t in targets}


def encode_labels(index: pd.DataFrame, label_maps: dict[str, list[str]]) -> dict[str, torch.Tensor]:
    """Integer labels per target; missing or unseen labels become ``IGNORE_INDEX``."""
    out = {}
    for target, classes in label_maps.items():
        lookup = {c: i for i, c in enumerate(classes)}
        out[target] = torch.tensor(
            [lookup.get(v, IGNORE_INDEX) if pd.notna(v) else IGNORE_INDEX for v in index[target]],
            dtype=torch.long,
        )
    return out


def save_label_maps(label_maps: dict[str, list[str]], path: str | Path) -> None:
    """Write label maps as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(label_maps, indent=2), encoding="utf-8")


def stratified_subset(index: pd.DataFrame, fraction: float, seed: int, col: str) -> np.ndarray:
    """Positions of a stratified subset (keeps at least one row per class)."""
    if fraction >= 1.0:
        return np.arange(len(index))
    sampled = index.groupby(col, group_keys=False).apply(
        lambda g: g.sample(max(1, math.ceil(len(g) * fraction)), random_state=seed)
    )
    return np.sort(index.index.get_indexer(sampled.index))


def class_weights(labels: torch.Tensor, n_classes: int, power: float) -> torch.Tensor:
    """Inverse-frequency class weights softened by ``power`` and normalised to mean 1.

    ``weight_c ~ (1 / count_c) ** power``. power=1 is full balancing (rare classes can dominate the
    loss), power=0 is no weighting; 0.5 is a middle ground that is easy to explain.
    """
    valid = labels[labels != IGNORE_INDEX]
    counts = torch.bincount(valid, minlength=n_classes).float().clamp(min=1)
    weights = counts.pow(-power)
    return weights * (n_classes / weights.sum())


def to_tensor(batch_u8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(np.array(batch_u8)).permute(0, 3, 1, 2).float() / 255.0


def normalise(x: torch.Tensor) -> torch.Tensor:
    """ImageNet mean/std normalisation for a (N, 3, H, W) tensor in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def resize(x: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize of a batch to the network input size."""
    return F.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)


def augment(x: torch.Tensor, cfg: dict[str, Any], generator: torch.Generator) -> torch.Tensor:
    """Photo-safe batch augmentation on (N, 3, H, W) floats in [0, 1].

    Horizontal flip, small rotation/scale/translation (one affine warp) and brightness, contrast and
    saturation jitter. Hue is NOT shifted (``cfg['hue']`` must be 0): rotating hue would turn a navy
    shirt into a purple one and silently corrupt the baseColour label.
    """
    if cfg.get("hue", 0.0) != 0.0:
        raise ValueError("hue augmentation would corrupt colour labels; keep augmentation.hue = 0")
    n = x.shape[0]

    def rnd(low: float, high: float) -> torch.Tensor:
        return low + (high - low) * torch.rand(n, generator=generator)

    flip = (torch.rand(n, generator=generator) < cfg["hflip_p"]).float() * -2 + 1  # -1 or +1
    angle = rnd(-cfg["rotation_deg"], cfg["rotation_deg"]) * math.pi / 180
    scale = rnd(*cfg["scale"])
    tx, ty = rnd(-cfg["translate"], cfg["translate"]), rnd(-cfg["translate"], cfg["translate"])
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack(
        [
            torch.stack([cos * flip, -sin, tx], dim=1),
            torch.stack([sin * flip, cos, ty], dim=1),
        ],
        dim=1,
    )
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)

    b = rnd(1 - cfg["brightness"], 1 + cfg["brightness"]).view(n, 1, 1, 1)
    c = rnd(1 - cfg["contrast"], 1 + cfg["contrast"]).view(n, 1, 1, 1)
    s = rnd(1 - cfg["saturation"], 1 + cfg["saturation"]).view(n, 1, 1, 1)
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * s + gray
    return x.clamp(0, 1)
