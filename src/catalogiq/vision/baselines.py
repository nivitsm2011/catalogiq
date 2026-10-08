"""Baselines the main model must beat.

1. Majority class (always guess the most common training label).
2. Frozen pretrained MobileNetV3 embeddings + one logistic-regression head per target.
3. Zero-shot CLIP: score each image against text prompts such as "a product photo of a t-shirt".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.utils import configure_cache_dirs
from catalogiq.vision.data import IGNORE_INDEX, normalise, resize, to_tensor

logger = get_logger(__name__)


@torch.no_grad()
def extract_backbone_embeddings(
    net: torch.nn.Module, images_u8: np.ndarray, hw: tuple[int, int], batch_size: int
) -> np.ndarray:
    """Frozen-backbone pooled features for every cached image (eval mode, no augmentation)."""
    net.eval()
    out = []
    for start in tqdm(range(0, len(images_u8), batch_size), desc="embeddings", mininterval=30):
        x = normalise(resize(to_tensor(images_u8[start : start + batch_size]), hw))
        out.append(net.embed(x).numpy())
    return np.concatenate(out)


def majority_scores(
    y_train: torch.Tensor, y_test: torch.Tensor, n_classes: int
) -> dict[str, float]:
    """Accuracy and macro-F1 of always predicting the most common training class."""
    from sklearn.metrics import f1_score

    valid_tr = y_train[y_train != IGNORE_INDEX]
    majority = int(torch.bincount(valid_tr, minlength=n_classes).argmax())
    mask = y_test != IGNORE_INDEX
    y_true = y_test[mask].numpy()
    pred = np.full_like(y_true, majority)
    return {
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
    }


def fit_logreg_heads(
    emb_train: np.ndarray,
    y_train: dict[str, torch.Tensor],
    cfg: dict[str, Any],
    seed: int,
) -> tuple[StandardScaler, dict[str, LogisticRegression]]:
    """One multinomial logistic regression per target on standardised embeddings."""
    scaler = StandardScaler().fit(emb_train)
    x = scaler.transform(emb_train)
    heads = {}
    for target, y in y_train.items():
        mask = (y != IGNORE_INDEX).numpy()
        clf = LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=seed)
        clf.fit(x[mask], y[mask].numpy())
        heads[target] = clf
        logger.info("logreg head fitted: %s (%d classes)", target, len(clf.classes_))
    return scaler, heads


def logreg_probabilities(
    scaler: StandardScaler, head: LogisticRegression, emb: np.ndarray, n_classes: int
) -> np.ndarray:
    """Class probabilities with all ``n_classes`` columns (absent classes get probability 0)."""
    probs = head.predict_proba(scaler.transform(emb))
    full = np.zeros((len(emb), n_classes), dtype=np.float32)
    full[:, head.classes_] = probs
    return full


def clip_zero_shot(
    images_u8: np.ndarray,
    label_maps: dict[str, list[str]],
    prompts: dict[str, str],
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Zero-shot CLIP class probabilities per target for the given images.

    Returns:
        target -> (N, n_classes) softmax probabilities over that target's prompts.
    """
    configure_cache_dirs()
    import open_clip

    cfg = get_config()["models"]["clip"]
    model, _, preprocess = open_clip.create_model_and_transforms(
        cfg["name"], pretrained=cfg["pretrained"]
    )
    tokenizer = open_clip.get_tokenizer(cfg["name"])
    model.eval()
    text_feats = {}
    with torch.no_grad():
        for target, classes in label_maps.items():
            texts = [prompts[target].format(c.lower()) for c in classes]
            f = model.encode_text(tokenizer(texts))
            text_feats[target] = f / f.norm(dim=-1, keepdim=True)
        logits: dict[str, list[np.ndarray]] = {t: [] for t in label_maps}
        for start in tqdm(range(0, len(images_u8), batch_size), desc="clip", mininterval=30):
            batch = torch.stack(
                [preprocess(Image.fromarray(a)) for a in images_u8[start : start + batch_size]]
            )
            img = model.encode_image(batch)
            img = img / img.norm(dim=-1, keepdim=True)
            for target, tf in text_feats.items():
                logits[target].append((100.0 * img @ tf.T).softmax(dim=-1).numpy())
    return {t: np.concatenate(v) for t, v in logits.items()}


def baseline_paths() -> tuple[Path, Path]:
    """Cache locations for the MobileNet and CLIP embeddings."""
    b = get_vision_config()["baselines"]
    return resolve_path(b["embeddings_path"]), resolve_path(b["clip_embeddings_path"])


def table_row(name: str, target: str, scores: dict[str, float]) -> dict[str, Any]:
    """One row of the baseline comparison table."""
    return {"model": name, "target": target, **{k: round(v, 4) for k, v in scores.items()}}


def to_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> DataFrame."""
    return pd.DataFrame(rows)
