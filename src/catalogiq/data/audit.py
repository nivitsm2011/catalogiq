"""Data audit for the fashion catalog: image health, duplicates, missing values, class counts."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from catalogiq.data.fashion import image_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def _inspect_image(path: Path, min_side: int) -> dict:
    """Open one image and return its health flags and perceptual hash."""
    if not path.exists():
        return {"image_exists": False, "image_ok": False, "phash": None, "width": 0, "height": 0}
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert("RGB")
            width, height = img.size
            phash = str(imagehash.phash(img))
        ok = min(width, height) >= min_side
        return {
            "image_exists": True,
            "image_ok": ok,
            "phash": phash,
            "width": width,
            "height": height,
        }
    except Exception as exc:  # noqa: BLE001 - any decode failure means "unreadable"
        logger.debug("unreadable image %s: %s", path, exc)
        return {"image_exists": True, "image_ok": False, "phash": None, "width": 0, "height": 0}


def inspect_images(
    df: pd.DataFrame, images_dir: str | Path, min_side: int, workers: int = 8
) -> pd.DataFrame:
    """Check every product image (exists, decodes, large enough) and compute its pHash.

    Args:
        df: Styles table with an ``id`` column.
        images_dir: Folder holding ``<id>.jpg`` files.
        min_side: Smallest acceptable image side in pixels.
        workers: Thread count for parallel decoding.

    Returns:
        ``df`` with ``image_exists``, ``image_ok``, ``phash``, ``width``, ``height`` columns.
    """
    paths = [image_path(images_dir, i) for i in df["id"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(
            tqdm(
                pool.map(lambda p: _inspect_image(p, min_side), paths),
                total=len(paths),
                desc="images",
            )
        )
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


def _hash_to_int(hex_hash: str) -> int:
    return int(hex_hash, 16)


def near_duplicate_groups(phashes: pd.Series, max_distance: int) -> pd.Series:
    """Group images whose perceptual hashes differ by at most ``max_distance`` bits.

    Uses pigeonhole bucketing: split each 64-bit hash into ``max_distance + 1`` chunks; two
    hashes within ``max_distance`` bits must share at least one identical chunk, so only
    hashes sharing a chunk are compared. Groups are connected components (union-find).

    Args:
        phashes: Hex pHash strings (NaN for unreadable images).
        max_distance: Maximum Hamming distance counted as "same photo".

    Returns:
        Integer group id per row (``-1`` where there is no hash); singletons get their own id.
    """
    valid = phashes.dropna()
    idx = valid.index.to_numpy()
    values = np.array([_hash_to_int(h) for h in valid], dtype=np.uint64)
    n = len(values)
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    n_chunks = max_distance + 1
    bounds = np.linspace(0, 64, n_chunks + 1).astype(int)
    for c in range(n_chunks):
        lo, hi = bounds[c], bounds[c + 1]
        mask = np.uint64((1 << (hi - lo)) - 1)
        keys = (values >> np.uint64(lo)) & mask
        buckets: dict[int, list[int]] = defaultdict(list)
        for i, key in enumerate(keys.tolist()):
            buckets[key].append(i)
        for members in buckets.values():
            if len(members) < 2:
                continue
            for a_pos, a in enumerate(members):
                for b in members[a_pos + 1 :]:
                    if find(a) != find(b) and bin(int(values[a]) ^ int(values[b])).count("1") <= (
                        max_distance
                    ):
                        parent[find(a)] = find(b)
    roots = np.array([find(i) for i in range(n)])
    _, group_ids = np.unique(roots, return_inverse=True)
    out = pd.Series(-1, index=phashes.index, dtype="int64")
    out.loc[idx] = group_ids
    return out


def audit_summary(df: pd.DataFrame, targets: list[str], load_report: dict) -> dict:
    """Compute and log every headline audit number.

    Args:
        df: Styles table after ``inspect_images`` and with ``dup_group`` assigned.
        targets: Target column names for class counts.
        load_report: ``LoadReport.as_dict()`` from the CSV parse.

    Returns:
        JSON-serialisable summary dictionary.
    """
    group_sizes = df.loc[df["dup_group"] >= 0].groupby("dup_group").size()
    summary = {
        "csv_parse": load_report,
        "rows": int(len(df)),
        "missing_per_column": {c: int(v) for c, v in df.isna().sum().items()},
        "duplicate_ids": int(df["id"].duplicated().sum()),
        "image_missing": int((~df["image_exists"]).sum()),
        "image_unreadable_or_tiny": int((df["image_exists"] & ~df["image_ok"]).sum()),
        "near_duplicate_groups_size_ge2": int((group_sizes >= 2).sum()),
        "images_in_near_duplicate_groups": int(group_sizes[group_sizes >= 2].sum()),
        "largest_near_duplicate_group": int(group_sizes.max()) if len(group_sizes) else 0,
        "class_counts": {t: df[t].value_counts(dropna=False).to_dict() for t in targets},
    }
    for key, value in summary.items():
        if key != "class_counts":
            logger.info("audit | %s = %s", key, value)
    for t in targets:
        logger.info("audit | %s: %d classes", t, df[t].nunique())
    return summary
