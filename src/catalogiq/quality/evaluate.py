"""Precision / recall helpers for the quality checks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config

CHECK_FOR_TYPE = {
    "blur": "blurry",
    "dark": "too_dark",
    "bright": "overexposed",
    "lowres": "low_resolution",
    "aspect": "bad_aspect_ratio",
    "clutter": "cluttered_background",
    "text": "text_or_watermark",
}
TYPE_FOR_CHECK = {v: k for k, v in CHECK_FOR_TYPE.items()}


def precision_recall(flagged: np.ndarray, positive: np.ndarray) -> dict[str, float]:
    """Precision, recall and F1 of boolean ``flagged`` against boolean ``positive``."""
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def per_check_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-check precision and recall on clean photos plus degraded copies.

    ``df`` has ``kind`` (clean / degraded), ``type``, ``level`` and one boolean column per check.
    A copy is a real defect for its own check only from ``positive_from_level`` on; milder copies of
    that type are left out of that check's scoring (neither right nor wrong). Every other photo,
    clean or degraded in a different way, counts as "no defect" for that check.
    """
    start = get_quality_config()["degrade"]["positive_from_level"]
    rows = []
    for check, dtype in TYPE_FOR_CHECK.items():
        own = (df["kind"] == "degraded") & (df["type"] == dtype)
        positive = (own & (df["level"] >= start[dtype])).to_numpy()
        keep = ~(own & (df["level"] < start[dtype])).to_numpy()
        res = precision_recall(df[check].to_numpy(bool)[keep], positive[keep])
        clean = (df["kind"] == "clean").to_numpy()
        res["false_alarm_rate_clean"] = float(df[check].to_numpy(bool)[clean].mean())
        rows.append({"check": check, **res})
    return pd.DataFrame(rows).set_index("check")


def recall_by_level(df: pd.DataFrame) -> pd.DataFrame:
    """Share of degraded copies flagged by their own check, by defect type and severity level."""
    out = []
    for dtype, check in CHECK_FOR_TYPE.items():
        sub = df[(df["kind"] == "degraded") & (df["type"] == dtype)]
        for lv, g in sub.groupby("level"):
            out.append(
                {"type": dtype, "level": int(lv), "recall": float(g[check].mean()), "n": len(g)}
            )
    return pd.DataFrame(out)
