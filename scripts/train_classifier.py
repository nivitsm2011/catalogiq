"""Train the multi-output attribute classifier (two stages) and save logits for evaluation.

Usage:
  python scripts/train_classifier.py --subset-fraction 0.2 --epochs1 1 --epochs2 1 --tag trial
  python scripts/train_classifier.py --tag full
Outputs: models/attribute_classifier/{best_stage1,best}.pt, data/interim/logits_<tag>.pt,
reports/training_history_<tag>.csv, MLflow run in mlruns/
"""

from __future__ import annotations

import argparse

import mlflow
import numpy as np
import pandas as pd
import torch

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.tracking import start_run
from catalogiq.utils import configure_cache_dirs, set_seed, timed
from catalogiq.vision.data import (
    build_image_cache,
    build_label_maps,
    encode_labels,
    save_label_maps,
    stratified_subset,
)
from catalogiq.vision.model import MultiHeadNet
from catalogiq.vision.train import build_class_weights, predict_logits, run_stage

logger = get_logger(__name__)


@timed
def main() -> None:
    """Train, then store validation and test logits for the evaluation notebook."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-fraction", type=float, default=None)
    parser.add_argument("--epochs1", type=int, default=None)
    parser.add_argument("--epochs2", type=int, default=None)
    parser.add_argument("--tag", default="full")
    parser.add_argument(
        "--skip-stage1",
        action="store_true",
        help="resume: load best_stage1_<tag>.pt and run stage 2 only",
    )
    args = parser.parse_args()

    configure_cache_dirs()
    cfg, v = get_config(), get_vision_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    images, index = build_image_cache()
    images = np.ascontiguousarray(images)  # load fully into RAM (609 MB) for fast batch indexing
    targets = v["targets"]
    label_maps = build_label_maps(index, targets)
    save_label_maps(label_maps, resolve_path(v["label_maps"]))
    labels = encode_labels(index, label_maps)
    split = index["split"].to_numpy()
    train_pos_all = np.where(split == "train")[0]
    val_pos, test_pos = np.where(split == "val")[0], np.where(split == "test")[0]

    fraction = args.subset_fraction or v["training"]["subset_fraction"]
    train_df = index.iloc[train_pos_all].reset_index(drop=True)
    train_pos = train_pos_all[stratified_subset(train_df, fraction, seed, "articleType")]
    logger.info(
        "train rows: %d (fraction %.2f), val %d, test %d",
        len(train_pos),
        fraction,
        len(val_pos),
        len(test_pos),
    )

    n_classes = {t: len(label_maps[t]) for t in targets}
    weights = build_class_weights(labels, train_pos, n_classes, v["training"]["class_weight_power"])
    net = MultiHeadNet(
        n_classes,
        feature_dim=v["backbone"]["feature_dim"],
        dropout=v["backbone"]["head_dropout"],
        weights=v["backbone"]["weights"],
    )
    ckpt_dir = resolve_path(v["training"]["checkpoint_dir"])
    params = {
        "tag": args.tag,
        "subset_fraction": fraction,
        "input_hw": v["input_hw"],
        "batch_size": v["batch_size"],
        "seed": seed,
        "backbone": v["backbone"]["name"],
        "unfreeze_from_block": v["backbone"]["unfreeze_from_block"],
        "class_weight_power": v["training"]["class_weight_power"],
        **{f"aug_{k}": x for k, x in v["augmentation"].items()},
        **{f"stage1_{k}": x for k, x in v["training"]["stage1"].items()},
        **{f"stage2_{k}": x for k, x in v["training"]["stage2"].items()},
    }
    with start_run(f"classifier_{args.tag}", params):
        stage1_ckpt = ckpt_dir / f"best_stage1_{args.tag}.pt"
        if args.skip_stage1:
            net.load_state_dict(torch.load(stage1_ckpt))
            logger.info("resumed from %s; running stage 2 only", stage1_ckpt.name)
            h1 = []
        else:
            h1 = run_stage(
                net,
                "stage1",
                images,
                train_pos,
                val_pos,
                labels,
                weights,
                v,
                seed,
                stage1_ckpt,
                args.epochs1,
            )
        h2 = run_stage(
            net,
            "stage2",
            images,
            train_pos,
            val_pos,
            labels,
            weights,
            v,
            seed,
            ckpt_dir / f"best_{args.tag}.pt",
            args.epochs2,
        )
        history = pd.DataFrame(h1 + h2)
        history.to_csv(resolve_path(f"reports/training_history_{args.tag}.csv"), index=False)
        hw = tuple(v["input_hw"])
        logits = {
            "val": predict_logits(net, images, val_pos, hw, v["batch_size"]),
            "test": predict_logits(net, images, test_pos, hw, v["batch_size"]),
        }
        torch.save(logits, resolve_path(f"data/interim/logits_{args.tag}.pt"))
        mlflow.log_artifact(str(resolve_path(f"reports/training_history_{args.tag}.csv")))
        logger.info(
            "mean seconds/epoch: stage1 %.0f, stage2 %.0f",
            np.mean([r["seconds"] for r in h1]),
            np.mean([r["seconds"] for r in h2]),
        )
        logger.info("best val macro-F1: %.4f", history["val_macro_f1"].max())


if __name__ == "__main__":
    main()
