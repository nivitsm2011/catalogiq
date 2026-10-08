"""Calibration and "not sure" thresholds.

Temperature scaling divides logits by one learned number T (fitted on the VALIDATION split) so that
confidences better match real accuracy. A per-target confidence threshold then decides when the app
asks the seller to confirm instead of auto-filling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from catalogiq.vision.data import IGNORE_INDEX


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find T > 0 minimising the validation negative log-likelihood of ``softmax(logits / T)``."""
    mask = labels != IGNORE_INDEX
    logits, labels = logits[mask], labels[mask]
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> np.ndarray:
    """Softmax probabilities of temperature-scaled logits."""
    return F.softmax(logits / temperature, dim=-1).numpy()


def choose_threshold(
    y: np.ndarray, probs: np.ndarray, target_precision: float, min_threshold: float = 0.0
) -> dict[str, float]:
    """Smallest confidence threshold whose accepted predictions reach ``target_precision``.

    Scans candidate thresholds (every observed confidence) from low to high and returns the first
    where accuracy-among-accepted >= ``target_precision``, which maximises coverage (the share of
    products auto-filled) subject to that quality bar. If no threshold reaches it, returns a
    threshold above 1.0 (always "needs review") and coverage 0.
    """
    mask = y != IGNORE_INDEX
    y, probs = y[mask], probs[mask]
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    order = np.argsort(conf)
    conf_sorted, correct_sorted = conf[order], correct[order]
    # accuracy of everything with confidence >= conf_sorted[i]
    suffix_correct = np.cumsum(correct_sorted[::-1])[::-1]
    suffix_count = np.arange(len(conf_sorted), 0, -1)
    precision_at = suffix_correct / suffix_count
    ok = np.where((precision_at >= target_precision) & (conf_sorted >= min_threshold))[0]
    if len(ok) == 0:
        return {"threshold": 1.01, "coverage": 0.0, "precision": float("nan")}
    i = int(ok[0])
    return {
        "threshold": float(conf_sorted[i]),
        "coverage": float(suffix_count[i] / len(conf_sorted)),
        "precision": float(precision_at[i]),
    }


def apply_threshold(y: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    """Coverage and precision of accepted predictions on a (test) split for a fixed threshold."""
    mask = y != IGNORE_INDEX
    y, probs = y[mask], probs[mask]
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    if not accepted.any():
        return {"coverage": 0.0, "precision": float("nan")}
    correct = probs.argmax(axis=1) == y
    return {"coverage": float(accepted.mean()), "precision": float(correct[accepted].mean())}
