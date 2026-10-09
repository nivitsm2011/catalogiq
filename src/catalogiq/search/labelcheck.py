"""Label check: does the catalog's articleType agree with the Phase 2 classifier?

The catalog contains plainly wrong labels (Phase 2 error analysis: about 10% of product-type
"mistakes" were wrong labels, e.g. a man in a vest labelled "Casual Shoes"). Recommending such
items would put a wrong photo into an outfit, so ``shop_the_look`` only uses items whose label the
classifier confirms.

Caveat: for train-split photos the classifier has seen the label, so the check is optimistic there;
it catches clear mislabels the model could not memorise, not every error.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from catalogiq.config import get_vision_config, resolve_path
from catalogiq.utils import configure_cache_dirs
from catalogiq.vision.data import build_image_cache, normalise, resize, to_tensor
from catalogiq.vision.model import MultiHeadNet

CACHE = "data/processed/label_check_articleType.parquet"


def compute_label_check(force: bool = False) -> pd.DataFrame:
    """Classifier prediction of articleType for every catalog photo (cached).

    Returns:
        Frame with ``id``, ``pred_articleType``, ``pred_confidence`` (softmax, uncalibrated).
    """
    path = resolve_path(CACHE)
    if path.exists() and not force:
        return pd.read_parquet(path)
    configure_cache_dirs()
    v = get_vision_config()
    model_dir = resolve_path(v["inference"]["model_dir"])
    labels = json.loads((model_dir / v["inference"]["labels_file"]).read_text(encoding="utf-8"))
    net = MultiHeadNet({t: len(c) for t, c in labels.items()}, pretrained=False)
    net.load_state_dict(torch.load(model_dir / "best_full.pt"))
    net.eval()
    images, index = build_image_cache()
    hw = tuple(v["input_hw"])
    classes = np.array(labels["articleType"])
    preds, confs = [], []
    with torch.no_grad():
        for b in tqdm(range(0, len(images), 256), desc="label check", mininterval=30):
            x = normalise(resize(to_tensor(images[b : b + 256]), hw))
            probs = torch.softmax(net.heads["articleType"](net.embed(x)), dim=-1)
            conf, idx = probs.max(dim=-1)
            preds.append(classes[idx.numpy()])
            confs.append(conf.numpy())
    out = pd.DataFrame(
        {
            "id": index["id"].to_numpy(),
            "pred_articleType": np.concatenate(preds),
            "pred_confidence": np.concatenate(confs).astype(np.float32),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return out


def verified_mask(catalog: pd.DataFrame, check: pd.DataFrame, min_confidence: float) -> np.ndarray:
    """Per catalog row: classifier agrees with the catalog articleType at this confidence."""
    merged = catalog[["id", "articleType"]].merge(check, on="id", how="left")
    ok = (merged["articleType"] == merged["pred_articleType"]) & (
        merged["pred_confidence"] >= min_confidence
    )
    return ok.to_numpy()
