"""Log the Phase 6 headline metrics to the local MLflow store (reads saved reports).

Usage: python scripts/log_quality_to_mlflow.py   (after the evaluate_* and run_quality_report runs)
"""

from __future__ import annotations

import json

import mlflow
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.quality.evaluate import per_check_scores
from catalogiq.tracking import start_run


def main() -> None:
    """Log per-check image metrics, consistency metrics and batch-report rates."""
    cfg = get_quality_config()
    out = resolve_path(cfg["paths"]["eval_dir"])
    metrics: dict[str, float] = {}
    scores = per_check_scores(pd.read_csv(out / "image_eval_flags.csv"))
    for check, row in scores.iterrows():
        metrics[f"{check}_recall"] = float(row["recall"])
        metrics[f"{check}_precision"] = float(row["precision"])
        metrics[f"{check}_clean_flag_rate"] = float(row["false_alarm_rate_clean"])
    cons = json.loads((out / "consistency_eval.json").read_text(encoding="utf-8"))
    for attr, row in cons["test"].items():
        metrics[f"consistency_{attr}_precision"] = row["precision"]
        metrics[f"consistency_{attr}_recall"] = row["recall"]
    summary = json.loads(resolve_path(cfg["paths"]["batch_summary"]).read_text(encoding="utf-8"))
    for key in ("pct_with_any_issue", "pct_needs_attention", "score_mean"):
        metrics[f"batch_{key}"] = summary[key]
    with start_run("phase6_quality_checker", {"consistency_mode": cons["chosen"]["mode"]}):
        mlflow.log_metrics(metrics)


if __name__ == "__main__":
    main()
