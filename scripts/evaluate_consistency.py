"""Image-text consistency: tune on validation with 300 injected mismatches, report on test.

Per split: 100 wrong colours, 100 wrong product types, 100 Men<->Women swaps are injected into the
stated attributes (photos are untouched). Thresholds and the way the two signals are combined are
chosen by F1 on the validation split; the test split is scored once. Precision is measured against
the injected rows only, so listings whose catalog label is genuinely wrong count as false alarms
(a lower bound); a sample of them is saved for manual review.
Usage: python scripts/evaluate_consistency.py
Outputs: reports/phase6/consistency_thresholds.json, consistency_eval.json and
consistency_natural_flags.csv
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.quality.consistency import (
    classifier_probabilities,
    clip_probabilities,
    flag_mismatch,
    inject_mismatches,
    mismatch_signals,
    value_embeddings,
)
from catalogiq.quality.evaluate import precision_recall
from catalogiq.search.embeddings import ClipEncoder
from catalogiq.vision.predict import AttributePredictor

logger = get_logger("evaluate_consistency")


def split_data(
    split: str, predictor: AttributePredictor, clip_img: np.ndarray, index: pd.DataFrame
):
    """Catalog rows, images, CLIP embeddings and classifier probabilities of one split (cached)."""
    cfg = get_quality_config()
    rows = np.where(index["split"].eq(split))[0]
    images = np.load(resolve_path(cfg["paths"]["image_cache"]), mmap_mode="r")
    cache = resolve_path(cfg["paths"]["classifier_probs_dir"])
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{split}.npz"
    if path.exists():
        probs = dict(np.load(path))
    else:
        probs = classifier_probabilities(np.asarray(images[rows]), predictor)
        np.savez_compressed(path, **probs)
    return index.iloc[rows].reset_index(drop=True), clip_img[rows], probs


def signals_for(table, attr, labels, clip_emb, clf, value_emb) -> pd.DataFrame:
    """Signals for one attribute using the (possibly modified) stated column."""
    to_idx = {v: i for i, v in enumerate(labels[attr])}
    stated = table[f"stated_{attr}"].map(to_idx).to_numpy()
    return mismatch_signals(stated, clf[attr], clip_probabilities(clip_emb, value_emb[attr]))


def score(table, sig_by_attr, attrs, min_conf, margin, mode) -> dict:
    """Precision / recall per attribute and overall for one threshold setting."""
    out, all_flag, all_pos = {}, [], []
    for a in attrs:
        f = flag_mismatch(sig_by_attr[a], min_conf, margin, mode)
        pos = table[f"injected_{a}"].to_numpy()
        out[a] = precision_recall(f, pos)
        all_flag.append(f)
        all_pos.append(pos)
    out["overall"] = precision_recall(np.concatenate(all_flag), np.concatenate(all_pos))
    return out


def main() -> None:
    """Tune on validation, evaluate on test."""
    cfg = get_quality_config()
    ccfg = cfg["consistency"]
    attrs = ccfg["attributes"]
    n_per = ccfg["n_injected"] // len(attrs)
    out_dir = resolve_path(cfg["paths"]["eval_dir"])
    index = pd.read_parquet(resolve_path(cfg["paths"]["image_index"]))
    index = index.merge(
        pd.read_parquet(resolve_path("data/processed/search_catalog.parquet"))[
            ["id", "productDisplayName"]
        ],
        on="id",
        how="left",
    )
    clip_img = np.load(resolve_path("data/processed/clip_image_emb.npy"))
    predictor = AttributePredictor()
    labels = predictor.labels
    encoder = ClipEncoder()
    value_emb = value_embeddings(encoder, labels)

    tables = {}
    for split, seed in (("val", cfg["seed"]), ("test", cfg["seed"] + 7)):
        base, emb, clf = split_data(split, predictor, clip_img, index)
        table = inject_mismatches(base, labels, n_per, np.random.default_rng(seed))
        sig = {a: signals_for(table, a, labels, emb, clf, value_emb) for a in attrs}
        tables[split] = (table, sig)
        logger.info(
            "%s: %d listings, injected %s",
            split,
            len(table),
            {a: int(table[f"injected_{a}"].sum()) for a in attrs},
        )

    table, sig = tables["val"]
    best = None
    grid_conf = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    grid_margin = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    for mode in ("clf", "clip", "or", "and"):
        for c in grid_conf if mode != "clip" else [0.5]:
            for m in grid_margin if mode != "clf" else [0.5]:
                f1 = score(table, sig, attrs, c, m, mode)["overall"]["f1"]
                if best is None or (f1 == f1 and f1 > best[0]):
                    best = (f1, {"min_conf": c, "margin": m, "mode": mode})
    chosen = best[1]
    logger.info("chosen on val: %s (F1 %.3f)", chosen, best[0])
    resolve_path(cfg["paths"]["consistency_thresholds"]).write_text(json.dumps(chosen, indent=1))

    t_table, t_sig = tables["test"]
    report = {"chosen": chosen, "val_f1": best[0], "test": {}, "test_variants": {}}
    report["test"] = score(
        t_table, t_sig, attrs, chosen["min_conf"], chosen["margin"], chosen["mode"]
    )
    # the single signals at settings tuned on val, for the baseline comparison
    for mode in ("clf", "clip", "or", "and"):
        sub = [
            (c, m, score(table, sig, attrs, c, m, mode)["overall"]["f1"])
            for c in (grid_conf if mode != "clip" else [0.5])
            for m in (grid_margin if mode != "clf" else [0.5])
        ]
        c, m, _ = max(sub, key=lambda x: (x[2] if x[2] == x[2] else -1))
        report["test_variants"][mode] = {
            "min_conf": c,
            "margin": m,
            **score(t_table, t_sig, attrs, c, m, mode)["overall"],
        }
    # natural flags (non-injected) for manual review
    flags = {
        a: flag_mismatch(t_sig[a], chosen["min_conf"], chosen["margin"], chosen["mode"])
        for a in attrs
    }
    natural = []
    for a in attrs:
        sel = np.where(
            flags[a]
            & ~t_table[f"injected_{a}"].to_numpy()
            & t_table[f"injected_{a}"].eq(False).to_numpy()
        )[0]
        for i in sel:
            if any(t_table[f"injected_{b}"].iat[i] for b in attrs):
                continue
            natural.append(
                {
                    "id": int(t_table["id"].iat[i]),
                    "attribute": a,
                    "stated": t_table[a].iat[i],
                    "classifier": labels[a][int(t_sig[a]["clf_top"].iat[i])],
                    "clf_conf": float(t_sig[a]["clf_alt_conf"].iat[i]),
                    "clip_best": labels[a][int(t_sig[a]["clip_best_alt"].iat[i])],
                    "clip_margin": float(t_sig[a]["clip_margin"].iat[i]),
                    "row": int(i),
                }
            )
    nat = pd.DataFrame(natural)
    nat.to_csv(out_dir / "consistency_natural_flags.csv", index=False)
    report["natural_flags"] = {
        "total": len(nat),
        "by_attribute": nat["attribute"].value_counts().to_dict() if len(nat) else {},
        "n_test": int(len(t_table)),
    }
    (out_dir / "consistency_eval.json").write_text(json.dumps(report, indent=1, default=float))
    logger.info("test overall: %s", report["test"]["overall"])


if __name__ == "__main__":
    main()
