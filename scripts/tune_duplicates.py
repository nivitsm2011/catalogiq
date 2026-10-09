"""Pick the duplicate-detection threshold from the hand-labelled pairs and report precision/recall.

Reads reports/duplicate_pairs_to_label.csv (pairs, similarity, pHash distance, sampling weight)
and reports/duplicate_labels.csv (pair_id, label in {1, 0, unsure}). "unsure" and blank rows are
dropped.

Rule for the recommended threshold: the lowest similarity at which the weighted precision of the
"similarity + pHash" detector is at least ``--min-precision`` (default 0.95). A false duplicate
can hide a legitimate listing, so precision is protected first and recall is then maximised. The
F1-optimal point is also reported.

Usage: python scripts/tune_duplicates.py [--min-precision 0.95]
Outputs: reports/duplicate_threshold_eval.csv, reports/duplicate_threshold.json,
         reports/duplicate_label_stats.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from catalogiq.config import get_search_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.search.duplicates import evaluate_thresholds

logger = get_logger(__name__)


def load_labelled(labels_path: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Merge pairs and labels; return the usable labelled frame and label statistics."""
    dcfg = get_search_config()["duplicates"]
    pairs = pd.read_csv(resolve_path(dcfg["pairs_csv"])).drop(columns=["label"], errors="ignore")
    labels_file = labels_path or dcfg["labels_csv"]
    labels = pd.read_csv(resolve_path(labels_file), dtype={"label": str})
    merged = pairs.merge(labels[["pair_id", "label"]], on="pair_id", how="left")
    raw = merged["label"].fillna("").str.strip().str.lower()
    stats = {
        "pairs_total": int(len(merged)),
        "duplicate": int((raw == "1").sum()),
        "not_duplicate": int((raw == "0").sum()),
        "unsure": int((raw == "unsure").sum()),
        "blank": int((raw == "").sum()),
    }
    usable = merged[raw.isin(["1", "0"])].assign(
        label=lambda d: d["label"].astype(str).str.strip().astype(int)
    )
    return usable.reset_index(drop=True), stats


def main() -> None:
    """Evaluate thresholds on the labelled sample and write the report files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--labels", default=None, help="labels CSV (default: config labels_csv)")
    args = parser.parse_args()
    dcfg = get_search_config()["duplicates"]
    labelled, stats = load_labelled(args.labels)
    stats["labels_file"] = args.labels or dcfg["labels_csv"]
    logger.info("label stats: %s", stats)
    if labelled["label"].nunique() < 2:
        raise SystemExit("need at least one duplicate and one non-duplicate label")

    limit = dcfg["phash_max_distance"]
    grid = np.round(np.arange(0.80, 1.0, 0.005), 4)
    table = evaluate_thresholds(labelled, grid, limit)
    table.to_csv(resolve_path("reports/duplicate_threshold_eval.csv"), index=False)

    # pHash alone, as a reference detector
    w, y = labelled["weight"].to_numpy(), labelled["label"].to_numpy().astype(bool)
    hash_rows = []
    for d in (0, 2, 4, 8, 12, 16):
        pred = (labelled["phash_distance"] <= d).to_numpy()
        tp, fp, fn = (w * (pred & y)).sum(), (w * (pred & ~y)).sum(), (w * (~pred & y)).sum()
        hash_rows.append(
            {
                "pHash max distance": d,
                "precision": round(float(tp / (tp + fp)) if tp + fp else float("nan"), 4),
                "recall": round(float(tp / (tp + fn)) if tp + fn else float("nan"), 4),
            }
        )
    hash_only = pd.DataFrame(hash_rows)
    logger.info("pHash alone:\n%s", hash_only.to_string(index=False))

    both = table[table["method"] == "similarity + pHash"].reset_index(drop=True)
    ok = both[both["precision"] >= args.min_precision]
    recommended = ok.iloc[0] if len(ok) else both.loc[both["f1"].idxmax()]
    f1_best = both.loc[both["f1"].idxmax()]
    result = {
        "recommended_threshold": float(recommended["threshold"]),
        "recommended_precision": float(recommended["precision"]),
        "recommended_recall": float(recommended["recall"]),
        "rule": (
            f"lowest threshold with weighted precision >= {args.min_precision} "
            f"(similarity + pHash <= {limit})"
        ),
        "precision_target_met": bool(len(ok)),
        "f1_optimal_threshold": float(f1_best["threshold"]),
        "f1_optimal_precision": float(f1_best["precision"]),
        "f1_optimal_recall": float(f1_best["recall"]),
        "phash_max_distance": limit,
        "pHash_only": hash_rows,
        "labels": stats,
    }
    resolve_path("reports/duplicate_threshold.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    resolve_path("reports/duplicate_label_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    logger.info("recommendation: %s", json.dumps(result))


if __name__ == "__main__":
    main()
