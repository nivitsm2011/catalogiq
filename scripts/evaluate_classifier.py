"""Evaluate the trained classifier on the held-out test split, calibrate, choose thresholds.

Usage: python scripts/evaluate_classifier.py --tag full
Inputs : data/interim/logits_<tag>.pt (from train_classifier.py), reports/baselines.csv
Outputs: reports/classifier_metrics.csv, reports/model_vs_baselines.csv, reports/calibration.csv,
         reports/per_class_<target>.csv, data/interim/test_probs_<tag>.npz,
         models/attribute_classifier/{labels,thresholds}.json
"""

from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np
import pandas as pd
import torch

from catalogiq.config import get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.tracking import start_run
from catalogiq.vision.calibration import (
    apply_temperature,
    apply_threshold,
    choose_threshold,
    fit_temperature,
)
from catalogiq.vision.data import build_image_cache, build_label_maps, encode_labels
from catalogiq.vision.evaluate import expected_calibration_error, per_class_table, score_target

logger = get_logger(__name__)


def main() -> None:
    """Score the model, fit temperatures on validation, pick thresholds, write all tables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    v = get_vision_config()
    ccfg = v["calibration"]
    _, index = build_image_cache()
    targets = v["targets"]
    label_maps = build_label_maps(index, targets)
    labels = encode_labels(index, label_maps)
    split = index["split"].to_numpy()
    val_pos, test_pos = np.where(split == "val")[0], np.where(split == "test")[0]
    logits = torch.load(resolve_path(f"data/interim/logits_{args.tag}.pt"))

    metric_rows, calib_rows, thresholds, test_probs = [], [], {}, {}
    with start_run(f"evaluation_{args.tag}", {"tag": args.tag, **ccfg}):
        for t in targets:
            y_val, y_test = labels[t][val_pos], labels[t][test_pos]
            temp = fit_temperature(logits["val"][t], y_val)
            raw_test = torch.softmax(logits["test"][t], dim=-1).numpy()
            cal_test = apply_temperature(logits["test"][t], temp)
            cal_val = apply_temperature(logits["val"][t], temp)
            yt = y_test.numpy()
            scores = score_target(yt, raw_test)
            ece_raw = expected_calibration_error(yt, raw_test, ccfg["n_bins"])
            ece_cal = expected_calibration_error(yt, cal_test, ccfg["n_bins"])
            chosen = choose_threshold(
                y_val.numpy(), cal_val, ccfg["target_precision"], ccfg["min_threshold"]
            )
            on_test = apply_threshold(yt, cal_test, chosen["threshold"])
            thresholds[t] = {
                "temperature": temp,
                "threshold": chosen["threshold"],
                "val_coverage": chosen["coverage"],
                "val_precision": chosen["precision"],
                "test_coverage": on_test["coverage"],
                "test_precision": on_test["precision"],
            }
            metric_rows.append({"target": t, **{k: round(x, 4) for k, x in scores.items()}})
            calib_rows.append(
                {
                    "target": t,
                    "temperature": round(temp, 3),
                    "ece_before": round(ece_raw, 4),
                    "ece_after": round(ece_cal, 4),
                    "threshold": round(chosen["threshold"], 4),
                    "test_coverage": round(on_test["coverage"], 4),
                    "test_precision_of_accepted": round(on_test["precision"], 4),
                }
            )
            per_class_table(yt, raw_test, label_maps[t]).round(4).to_csv(
                resolve_path(f"reports/per_class_{t}.csv"), index=False
            )
            test_probs[t] = cal_test
            mlflow.log_metrics({f"test_{t}_{k}": x for k, x in scores.items()})
            mlflow.log_metrics({f"test_{t}_ece_after": ece_cal, f"{t}_temperature": temp})

        metrics = pd.DataFrame(metric_rows)
        metrics.to_csv(resolve_path("reports/classifier_metrics.csv"), index=False)
        pd.DataFrame(calib_rows).to_csv(resolve_path("reports/calibration.csv"), index=False)
        np.savez_compressed(resolve_path(f"data/interim/test_probs_{args.tag}.npz"), **test_probs)

        base = pd.read_csv(resolve_path("reports/baselines.csv"))
        main_rows = metrics.assign(model="multi-output MobileNetV3 (fine-tuned)")
        both = pd.concat([base, main_rows], ignore_index=True)
        both.to_csv(resolve_path("reports/model_vs_baselines.csv"), index=False)
        mlflow.log_artifact(str(resolve_path("reports/model_vs_baselines.csv")))
        logger.info("test metrics:\n%s", metrics.to_string(index=False))
        logger.info("calibration:\n%s", pd.DataFrame(calib_rows).to_string(index=False))

    model_dir = resolve_path(v["inference"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / v["inference"]["labels_file"]).write_text(
        json.dumps(label_maps, indent=2), encoding="utf-8"
    )
    (model_dir / v["inference"]["thresholds_file"]).write_text(
        json.dumps(thresholds, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
