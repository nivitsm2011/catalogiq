"""Tune the image-check thresholds on the VALIDATION split and write reports/phase6/thresholds.json.

Rule (stated before looking at results): for each measure choose the threshold with the highest
recall
on real defects (degraded copies at or above ``positive_from_level`` plus the hand-labelled natural
defects) that still flags at most ``MAX_FPR`` of the CLEAN validation photos. Resolution
and aspect ratio are rules, not tuned. Test photos are never used here.
Usage: python scripts/tune_quality_thresholds.py
Outputs: reports/phase6/thresholds.json, reports/phase6/tuning_val.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.quality.clip_checks import prompt_embeddings, text_margin
from catalogiq.quality.degrade import make_degraded_set
from catalogiq.quality.image_checks import compute_metrics
from catalogiq.search.embeddings import ClipEncoder

logger = get_logger("tune_quality_thresholds")
N_CLEAN = 800
MAX_FPR = 0.03

# check -> (degradation type, metric, direction); "low" flags values below the threshold
TUNED = {
    "blur_min_laplacian_var": ("blur", "blur", "low"),
    "dark_max_mean_luma": ("dark", "mean_luma", "low"),
    "bright_min_p5_luma": ("bright", "p5_luma", "high"),
    "clutter_max_corner_std": ("clutter", "corner_std", "high"),
    "text_min_clip_margin": ("text", "text_margin", "high"),
}


def best_threshold(
    pos: np.ndarray, neg: np.ndarray, direction: str, max_fpr: float
) -> tuple[float, float, float]:
    """Threshold maximising recall subject to a false-positive rate cap on ``neg``.

    Returns:
        (threshold, recall, false-positive rate).
    """
    values = np.unique(np.concatenate([pos, neg]))
    best = (values[0], -1.0, 1.0)
    for t in values:
        fpr = float((neg < t).mean() if direction == "low" else (neg > t).mean())
        rec = float((pos < t).mean() if direction == "low" else (pos > t).mean())
        if fpr <= max_fpr and (rec > best[1] or (rec == best[1] and fpr < best[2])):
            best = (float(t), rec, fpr)
    return best


def main() -> None:
    """Measure clean, degraded and hand-labelled validation photos and pick the thresholds."""
    cfg = get_quality_config()
    rng = np.random.default_rng(cfg["seed"])
    images = np.load(resolve_path(cfg["paths"]["image_cache"]), mmap_mode="r")
    index = pd.read_parquet(resolve_path(cfg["paths"]["image_index"]))
    val = np.where(index["split"].eq("val"))[0]
    labels = pd.read_csv(resolve_path(cfg["paths"]["image_labels"]))
    labelled_rows = {
        int(i): r for i, r in zip(labels["id"], labels.to_dict("records"), strict=True)
    }
    id_to_row = dict(zip(index["id"].to_numpy(), range(len(index)), strict=True))
    lab_rows = [id_to_row[i] for i in labelled_rows]
    clean = [r for r in rng.choice(val, N_CLEAN, replace=False) if r not in lab_rows]
    pick = np.asarray(images[clean])
    rows = [{"kind": "clean", "type": "", "level": -1, **compute_metrics(a)} for a in pick]
    text_copies = []
    for _, t, lv, out in make_degraded_set(pick, rng):
        rows.append({"kind": "degraded", "type": t, "level": lv, **compute_metrics(out)})
        if t == "text":
            text_copies.append(out)
    for r in lab_rows:
        lab = labelled_rows[int(index["id"].iloc[r])]
        kind = "natural_defect" if lab["cluttered_background"] else "natural_ok"
        rows.append(
            {
                "kind": kind,
                "type": "clutter" if kind == "natural_defect" else "",
                "level": -1,
                **compute_metrics(np.asarray(images[r])),
            }
        )
    df = pd.DataFrame(rows)
    # CLIP text-overlay score: only needed for clean photos, the labelled sample and the text copies
    encoder = ClipEncoder()
    prompts = prompt_embeddings(encoder)
    # rows are ordered: clean, degraded (text copies in order), natural labelled
    need = df.index[(df["kind"] != "degraded") | (df["type"] == "text")]
    by_row = dict(zip(need[df.loc[need, "kind"].eq("clean").to_numpy()], pick, strict=True))
    by_row |= dict(
        zip(df.index[df["kind"].eq("degraded") & df["type"].eq("text")], text_copies, strict=True)
    )
    by_row |= dict(
        zip(
            df.index[df["kind"].str.startswith("natural")],
            [np.asarray(images[r]) for r in lab_rows],
            strict=True,
        )
    )
    arrays = [by_row[i] for i in need]
    assert len(need) == len(arrays)
    emb = np.concatenate(
        [encoder.encode_images_u8(np.stack(arrays[i : i + 64])) for i in range(0, len(arrays), 64)]
    )
    df["text_margin"] = np.nan
    df.loc[need, "text_margin"] = text_margin(emb, prompts)
    df.to_csv(resolve_path(cfg["paths"]["eval_dir"]) / "tuning_val.csv", index=False)
    start = cfg["degrade"]["positive_from_level"]
    negatives = df[df["kind"].isin(["clean", "natural_ok"])]
    tuned: dict[str, float] = {}
    report = {}
    for name, (dtype, metric, direction) in TUNED.items():
        is_pos = (
            (df["kind"] == "degraded") & (df["type"] == dtype) & (df["level"] >= start[dtype])
        ) | ((df["kind"] == "natural_defect") & (df["type"] == dtype))
        cap = MAX_FPR
        t, rec, fpr = best_threshold(
            df.loc[is_pos, metric].to_numpy(), negatives[metric].to_numpy(), direction, cap
        )
        tuned[name] = round(t, 4)
        report[name] = {
            "threshold": tuned[name],
            "recall_val": round(rec, 3),
            "fpr_val": round(fpr, 3),
            "n_pos": int(is_pos.sum()),
            "n_neg": int(len(negatives)),
        }
        logger.info("%s -> %s", name, report[name])
    resolve_path(cfg["paths"]["thresholds"]).write_text(
        json.dumps(tuned, indent=1), encoding="utf-8"
    )
    (resolve_path(cfg["paths"]["eval_dir"]) / "tuning_report.json").write_text(
        json.dumps(report, indent=1)
    )


if __name__ == "__main__":
    main()
