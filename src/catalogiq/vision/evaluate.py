"""Evaluation metrics shared by baselines and the main model (all on class probabilities)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from catalogiq.vision.data import IGNORE_INDEX


def _valid(y: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = y != IGNORE_INDEX
    return y[mask], probs[mask]


def top_k_accuracy(y: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Share of rows whose true class is among the k most probable."""
    k = min(k, probs.shape[1])
    top = np.argpartition(-probs, k - 1, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def score_target(y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Accuracy, macro-F1, weighted-F1 and top-3 accuracy for one target."""
    y, probs = _valid(y, probs)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "top3_accuracy": top_k_accuracy(y, probs, 3),
    }


def per_class_table(y: np.ndarray, probs: np.ndarray, classes: list[str]) -> pd.DataFrame:
    """Per-class precision, recall, F1 and support, sorted by support."""
    y, probs = _valid(y, probs)
    pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    table = pd.DataFrame({"class": classes, "precision": p, "recall": r, "f1": f, "support": s})
    return table.sort_values("support", ascending=False).reset_index(drop=True)


def confusion_top_k(
    y: np.ndarray, probs: np.ndarray, classes: list[str], k: int = 15
) -> tuple[np.ndarray, list[str]]:
    """Confusion matrix restricted to the ``k`` most frequent true classes (rows = true).

    Predictions outside the top-k classes are grouped into one extra column, "(other)".
    """
    y, probs = _valid(y, probs)
    pred = probs.argmax(axis=1)
    top = pd.Series(y).value_counts().index[:k].tolist()
    index = {c: i for i, c in enumerate(top)}
    rows = np.array([index[t] for t in y if t in index])
    cols = np.array([index.get(p, len(top)) for t, p in zip(y, pred, strict=True) if t in index])
    cm = confusion_matrix(rows, cols, labels=list(range(len(top) + 1)))[: len(top)]
    return cm, [classes[c] for c in top] + ["(other)"]


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """ECE: average |accuracy - confidence| over confidence bins, weighted by bin size."""
    y, probs = _valid(y, probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def reliability_bins(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Per-bin confidence vs accuracy (for the reliability diagram)."""
    y, probs = _valid(y, probs)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            rows.append(
                {
                    "bin": f"{lo:.1f}-{hi:.1f}",
                    "mean_confidence": float(conf[in_bin].mean()),
                    "accuracy": float(correct[in_bin].mean()),
                    "count": int(in_bin.sum()),
                }
            )
    return pd.DataFrame(rows)
