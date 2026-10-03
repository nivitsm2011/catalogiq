"""Group-aware stratified train/val/test splits.

Near-duplicate photos (same product shot twice, colour variants of one shot) must land in the same
split. If one copy is in train and its twin in test, the model is "tested" on an image it has
effectively memorised, which inflates accuracy and hides real-world error. We therefore split by
*group* (near-duplicate cluster) while keeping the articleType mix equal across splits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def make_splits(
    df: pd.DataFrame,
    stratify_col: str,
    group_col: str,
    fractions: dict[str, float],
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    """Assign each row to train/val/test with group-aware stratification.

    The data is cut into ``n_folds`` stratified, group-disjoint folds; whole folds are then
    assigned to splits so that the fractions are met (e.g. 20 folds -> 14/3/3 for 70/15/15).

    Args:
        df: Cleaned table.
        stratify_col: Column whose class mix should be preserved (articleType).
        group_col: Near-duplicate group id; a group never spans two splits.
        fractions: Mapping with ``train``, ``val``, ``test`` fractions.
        n_folds: Number of folds to cut (must make the fractions whole numbers of folds).
        seed: Random seed.

    Returns:
        Copy of ``df`` with a ``split`` column.
    """
    counts = {k: v * n_folds for k, v in fractions.items()}
    if any(abs(c - round(c)) > 1e-9 for c in counts.values()):
        raise ValueError(f"fractions {fractions} are not whole numbers of {n_folds} folds")
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(df, df[stratify_col], df[group_col])):
        fold[test_idx] = k
    order = ["test"] * round(counts["test"]) + ["val"] * round(counts["val"])
    order += ["train"] * (n_folds - len(order))
    out = df.copy()
    out["split"] = [order[f] for f in fold]
    return out


def verify_no_leakage(df: pd.DataFrame, group_col: str) -> int:
    """Return how many groups span more than one split (must be 0)."""
    spans = df.groupby(group_col)["split"].nunique()
    leaked = int((spans > 1).sum())
    logger.info("leakage check | groups spanning >1 split: %d", leaked)
    return leaked
