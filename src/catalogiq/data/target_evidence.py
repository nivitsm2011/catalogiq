"""Evidence for which targets are worth predicting from a photo.

A target should only be promised to sellers if a photo can determine it *consistently*. Three
simple, explainable numbers per target:

* ``majority_share``: how dominant the biggest class is (a model must beat this to add value);
* ``pct_missing``: how often the catalog itself leaves the field empty;
* ``twin_disagreement``: among groups of (near-)identical photos, the share of groups whose members
  carry *different* labels. The same picture should get the same label; disagreement is a direct
  measure of annotation noise, i.e. a ceiling on achievable accuracy. Caveat: twins can be
  colourways of one product, so colour disagreement is partly legitimate.

The real test is a trained baseline on held-out data (Phase 2); this is the pre-modelling signal.
"""

from __future__ import annotations

import pandas as pd

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def twin_disagreement(df: pd.DataFrame, target: str, group_col: str) -> tuple[float, int]:
    """Share of multi-image groups whose members disagree on ``target``.

    Returns:
        (disagreement rate, number of groups with at least 2 labelled images).
    """
    labelled = df.dropna(subset=[target])
    sizes = labelled.groupby(group_col)[target].transform("size")
    groups = labelled[sizes >= 2].groupby(group_col)[target].nunique()
    if groups.empty:
        return float("nan"), 0
    return float((groups > 1).mean()), int(len(groups))


def target_evidence(
    raw: pd.DataFrame, clean: pd.DataFrame, targets: list[str], group_col: str
) -> pd.DataFrame:
    """Build the per-target evidence table.

    Args:
        raw: Audited table before cleaning (for missing-value rates).
        clean: Cleaned table (for class balance and twin disagreement).
        targets: Targets to evaluate.
        group_col: Near-duplicate group id column.

    Returns:
        One row per target.
    """
    rows = []
    for target in targets:
        disagreement, n_groups = twin_disagreement(clean, target, group_col)
        shares = clean[target].value_counts(normalize=True)
        rows.append(
            {
                "target": target,
                "n_classes": int(clean[target].nunique()),
                "majority_share": round(float(shares.iloc[0]), 4),
                "pct_missing": round(100 * float(raw[target].isna().mean()), 3),
                "twin_groups": n_groups,
                "twin_disagreement": round(disagreement, 4),
            }
        )
    result = pd.DataFrame(rows)
    logger.info("target evidence:\n%s", result.to_string(index=False))
    return result
