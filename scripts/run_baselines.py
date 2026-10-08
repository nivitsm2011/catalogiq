"""Baselines for the attribute classifier: majority, frozen-embedding logistic regression, CLIP.

Usage: python scripts/run_baselines.py [--skip-clip]
Outputs: data/interim/emb_mobilenet.npz, data/interim/baseline_probs.npz, reports/baselines.csv
"""

from __future__ import annotations

import argparse
import json

import mlflow
import numpy as np

from catalogiq.config import get_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.tracking import start_run
from catalogiq.utils import configure_cache_dirs, set_seed, timed
from catalogiq.vision import baselines as bl
from catalogiq.vision.data import (
    build_image_cache,
    build_label_maps,
    encode_labels,
    save_label_maps,
)
from catalogiq.vision.evaluate import score_target
from catalogiq.vision.model import MultiHeadNet

logger = get_logger(__name__)


@timed
def main() -> None:
    """Run all baselines on the held-out test split and log them to MLflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-clip", action="store_true")
    args = parser.parse_args()
    configure_cache_dirs()
    cfg, v = get_config(), get_vision_config()
    seed = cfg["project"]["seed"]
    set_seed(seed)
    images, index = build_image_cache()
    targets = v["targets"]
    label_maps = build_label_maps(index, targets)
    save_label_maps(label_maps, resolve_path(v["label_maps"]))
    labels = encode_labels(index, label_maps)
    split = index["split"].to_numpy()
    tr, te = split == "train", split == "test"
    rows, probs_out = [], {}

    # --- frozen MobileNetV3 embeddings (cached) ---
    emb_path, clip_path = bl.baseline_paths()
    if emb_path.exists():
        emb = np.load(emb_path)["emb"]
    else:
        net = MultiHeadNet({t: len(label_maps[t]) for t in targets}, pretrained=True)
        emb = bl.extract_backbone_embeddings(net, images, tuple(v["input_hw"]), v["batch_size"])
        np.savez_compressed(emb_path, emb=emb)
    logger.info("embeddings: %s", emb.shape)

    with start_run(
        "baseline_majority_and_logreg", {"input_hw": v["input_hw"], **v["baselines"]["logreg"]}
    ):
        scaler, heads = bl.fit_logreg_heads(
            emb[tr], {t: labels[t][tr] for t in targets}, v["baselines"]["logreg"], seed
        )
        for t in targets:
            n = len(label_maps[t])
            maj = bl.majority_scores(labels[t][tr], labels[t][te], n)
            rows.append(bl.table_row("majority class", t, maj))
            p = bl.logreg_probabilities(scaler, heads[t], emb[te], n)
            probs_out[f"logreg_{t}"] = p
            sc = score_target(labels[t][te].numpy(), p)
            rows.append(bl.table_row("frozen MobileNetV3 + logistic regression", t, sc))
            mlflow.log_metrics({f"logreg_{t}_{k}": x for k, x in sc.items()})
            mlflow.log_metrics({f"majority_{t}_{k}": x for k, x in maj.items()})

    # --- zero-shot CLIP (test split only) ---
    if not args.skip_clip:
        with start_run("baseline_clip_zero_shot", {**get_config()["models"]["clip"]}):
            clip_probs = bl.clip_zero_shot(images[te], label_maps, v["baselines"]["clip_prompts"])
            for t in targets:
                probs_out[f"clip_{t}"] = clip_probs[t]
                sc = score_target(labels[t][te].numpy(), clip_probs[t])
                rows.append(bl.table_row("zero-shot CLIP ViT-B/32", t, sc))
                mlflow.log_metrics({f"clip_{t}_{k}": x for k, x in sc.items()})
            mlflow.log_dict(v["baselines"]["clip_prompts"], "clip_prompts.json")

    table = bl.to_frame(rows)
    table.to_csv(resolve_path("reports/baselines.csv"), index=False)
    np.savez_compressed(resolve_path("data/interim/baseline_probs.npz"), **probs_out)
    logger.info("baselines:\n%s", table.to_string(index=False))
    print(json.dumps({"rows": len(table)}))


if __name__ == "__main__":
    main()
