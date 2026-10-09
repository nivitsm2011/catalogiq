"""Build the 100-pair duplicate labelling sheet (a click-to-label HTML page + the pairs CSV).

Pairs are drawn from the CLIP nearest-neighbour candidates, stratified over similarity bins so the
sample contains clear duplicates, clear non-duplicates and the hard cases in between. Each pair gets
a sampling weight (bin population / bin sample size) so that a threshold chosen on 100 labels can
estimate precision and recall for the whole candidate population.

Usage: python scripts/make_labeling_sheet.py
Outputs: reports/duplicate_pairs_to_label.csv, reports/duplicate_labeling.html
"""

from __future__ import annotations

import base64
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_search_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.logging_utils import get_logger
from catalogiq.search.duplicates import DuplicateFinder, load_phash
from catalogiq.search.embeddings import embed_catalog

logger = get_logger(__name__)

TEMPLATE = Path(__file__).with_name("labeling_template.html")


def _thumb(pid: int, images_dir) -> str:
    raw = image_path(images_dir, pid).read_bytes()
    return "data:image/jpeg;base64," + base64.b64encode(raw).decode()


def main() -> None:
    """Sample 100 pairs across similarity bins and write the labelling page."""
    cfg, scfg = get_config(), get_search_config()
    dcfg = scfg["duplicates"]
    image_emb, _, catalog = embed_catalog()
    finder = DuplicateFinder(
        catalog, image_emb, load_phash(catalog["id"]), min_similarity=dcfg["similarity_bins"][0]
    )
    pairs = finder.candidate_pairs().copy()
    edges = dcfg["similarity_bins"]
    pairs["bin"] = pd.cut(pairs["similarity"], bins=edges, right=False).astype(str)
    n_total, bins = dcfg["labeled_pairs"], pairs["bin"].unique().tolist()
    per_bin = n_total // len(bins)
    rng = np.random.default_rng(cfg["project"]["seed"])
    picked = []
    for b in sorted(bins):
        group = pairs[pairs["bin"] == b]
        take = group.sample(min(per_bin, len(group)), random_state=int(rng.integers(1e9)))
        take = take.assign(weight=len(group) / max(len(take), 1))
        picked.append(take)
        logger.info("bin %s: population %d, sampled %d", b, len(group), len(take))
    sample = pd.concat(picked)
    if len(sample) < n_total:  # top up from the least-covered bins
        rest = pairs.drop(sample.index)
        extra = rest.sample(n_total - len(sample), random_state=int(rng.integers(1e9)))
        sample = pd.concat([sample, extra.assign(weight=1.0)])
    sample = sample.sample(frac=1, random_state=cfg["project"]["seed"]).reset_index(drop=True)
    # randomly swap A/B so position carries no information
    flip = np.random.default_rng(1).random(len(sample)) < 0.5
    a, b = sample["id_a"].copy(), sample["id_b"].copy()
    sample["id_a"], sample["id_b"] = np.where(flip, b, a), np.where(flip, a, b)
    sample.insert(0, "pair_id", np.arange(1, len(sample) + 1))
    sample["label"] = ""

    images_dir = resolve_path(cfg["data"]["fashion"]["images_dir"])
    payload = [
        {
            "pair_id": int(r.pair_id),
            "id_a": int(r.id_a),
            "id_b": int(r.id_b),
            "img_a": _thumb(int(r.id_a), images_dir),
            "img_b": _thumb(int(r.id_b), images_dir),
        }
        for r in sample.itertuples()
    ]
    resolve_path(dcfg["pairs_csv"]).parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(resolve_path(dcfg["pairs_csv"]), index=False)
    html = TEMPLATE.read_text(encoding="utf-8").replace("__PAIRS__", json.dumps(payload))
    resolve_path(dcfg["labeling_html"]).write_text(html, encoding="utf-8")
    random.seed(0)
    logger.info("wrote %d pairs; page %.1f MB", len(sample), len(html) / 1e6)


if __name__ == "__main__":
    main()
