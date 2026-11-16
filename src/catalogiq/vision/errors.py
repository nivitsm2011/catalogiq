"""Error analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.vision.data import IGNORE_INDEX


def most_confident_wrong(
    y: np.ndarray, probs: np.ndarray, classes: list[str], ids: np.ndarray, k: int = 24
) -> pd.DataFrame:
    """The ``k`` wrong predictions the model was most sure about.

    High-confidence mistakes are the best place to look for mislabelled products: a model that is
    very sure and disagrees with the catalog is often right and the label is wrong.
    """
    mask = y != IGNORE_INDEX
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    wrong = mask & (pred != y)
    idx = np.where(wrong)[0]
    idx = idx[np.argsort(-conf[idx])][:k]
    return pd.DataFrame(
        {
            "position": idx,
            "id": ids[idx],
            "true": [classes[y[i]] for i in idx],
            "predicted": [classes[pred[i]] for i in idx],
            "confidence": conf[idx].round(4),
        }
    )
