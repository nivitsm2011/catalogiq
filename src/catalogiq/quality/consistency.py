"""Image-text consistency: does the photo match what the listing says?

Two independent signals per stated attribute (colour, product type, gender):
- the Phase 2 classifier: it must be confident that the photo shows a DIFFERENT value;
- CLIP zero-shot: the probability CLIP gives the best alternative value beats the stated one by
  a margin.
Both signals read the photo only, never the listing text, so a wrong listing cannot hide it.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import torch

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.vision.predict import AttributePredictor

CLIP_TEMPERATURE = 100.0


def value_embeddings(encoder: Any, labels: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """CLIP embeddings of one prompt per value of each consistency attribute."""
    cfg = get_quality_config()["consistency"]
    out = {}
    for attr in cfg["attributes"]:
        prompts = [cfg["clip_prompts"][attr].format(value=v.lower()) for v in labels[attr]]
        out[attr] = encoder.encode_texts(prompts)
    return out


def clip_probabilities(image_emb: np.ndarray, value_emb: np.ndarray) -> np.ndarray:
    """Zero-shot probabilities over an attribute's values, (N, K)."""
    logits = CLIP_TEMPERATURE * image_emb @ value_emb.T
    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


@torch.no_grad()
def classifier_probabilities(
    images_u8: np.ndarray, predictor: AttributePredictor, batch: int = 128
) -> dict[str, np.ndarray]:
    """Calibrated class probabilities of the Phase 2 classifier for every attribute, (N, K) each."""
    chunks: dict[str, list[np.ndarray]] = {t: [] for t in predictor.targets}
    for i in range(0, len(images_u8), batch):
        x = torch.cat([predictor.preprocess(im) for im in images_u8[i : i + batch]])
        for target, lg in zip(predictor.targets, predictor.model(x), strict=True):
            chunks[target].append(torch.softmax(lg / predictor.temperature[target], dim=-1).numpy())
    return {t: np.concatenate(c) for t, c in chunks.items()}


def mismatch_signals(
    stated: np.ndarray, clf_probs: np.ndarray, clip_probs: np.ndarray
) -> pd.DataFrame:
    """Per listing: how strongly each signal says the stated value is wrong.

    Args:
        stated: Index of the stated value per listing (into the attribute's value list).
        clf_probs: Classifier probabilities (N, K).
        clip_probs: CLIP zero-shot probabilities (N, K).

    Returns:
        ``clf_alt_conf`` (confidence in the top value when it differs from the stated one, else 0),
        ``clip_margin`` (best alternative minus stated, CLIP), plus the predicted and CLIP-best
        indices.
    """
    n = np.arange(len(stated))
    top = clf_probs.argmax(axis=1)
    clip_alt = clip_probs.copy()
    clip_alt[n, stated] = -1.0
    best_alt = clip_alt.argmax(axis=1)
    return pd.DataFrame(
        {
            "clf_top": top,
            "clf_alt_conf": np.where(top != stated, clf_probs.max(axis=1), 0.0),
            "clip_best_alt": best_alt,
            "clip_margin": clip_alt[n, best_alt] - clip_probs[n, stated],
        }
    )


def flag_mismatch(sig: pd.DataFrame, min_conf: float, margin: float, mode: str) -> np.ndarray:
    """Boolean mismatch flag. ``mode``: ``clf``, ``clip``, ``or`` or ``and``."""
    clf = sig["clf_alt_conf"].to_numpy() >= min_conf
    clip = sig["clip_margin"].to_numpy() >= margin
    return {"clf": clf, "clip": clip, "or": clf | clip, "and": clf & clip}[mode]


def inject_mismatches(
    catalog: pd.DataFrame, labels: dict[str, list[str]], n_per_attr: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Copy of ``catalog`` with ``n_per_attr`` clearly wrong values per attribute.

    Colour: a different family that does not look alike (``similar_colours``). Product type: a type
    from a different sub-category. Gender: Men <-> Women (only listings stated as Men or Women).
    Returns the table with ``stated_<attr>`` columns and an ``injected_<attr>`` boolean per
    attribute.
    Each listing receives at most one injected mismatch.
    """
    cfg = get_quality_config()["consistency"]
    similar = {frozenset(p) for p in cfg["similar_colours"]}
    out = catalog.copy()
    for attr in cfg["attributes"]:
        out[f"stated_{attr}"] = out[attr]
        out[f"injected_{attr}"] = False
    taken: set[int] = set()
    sub_of = dict(zip(catalog["articleType"], catalog["subCategory"], strict=False))
    for attr in cfg["attributes"]:
        pool = [i for i in rng.permutation(len(out)) if i not in taken]
        done = 0
        for i in pool:
            cur = out[attr].iat[i]
            if attr == "gender":
                if cur not in ("Men", "Women"):
                    continue
                new = "Women" if cur == "Men" else "Men"
            elif attr == "baseColour":
                options = [
                    v for v in labels[attr] if v != cur and frozenset((v, cur)) not in similar
                ]
                if not options or cur == "Multi":
                    continue
                new = options[int(rng.integers(len(options)))]
            else:
                options = [v for v in labels[attr] if sub_of.get(v) not in (None, sub_of.get(cur))]
                if not options:
                    continue
                new = options[int(rng.integers(len(options)))]
            out.iat[i, out.columns.get_loc(f"stated_{attr}")] = new
            out.iat[i, out.columns.get_loc(f"injected_{attr}")] = True
            taken.add(i)
            done += 1
            if done == n_per_attr:
                break
    return out


def load_consistency_thresholds() -> dict[str, Any]:
    """Tuned ``min_conf``, ``margin`` and ``mode`` (config defaults until tuned)."""
    cfg = get_quality_config()["consistency"]
    out = {
        "min_conf": cfg["min_classifier_conf"],
        "margin": cfg["clip_min_alt_margin"],
        "mode": "or",
    }
    path = resolve_path(get_quality_config()["paths"]["consistency_thresholds"])
    if path.exists():
        out.update(json.loads(path.read_text(encoding="utf-8")))
    return out
