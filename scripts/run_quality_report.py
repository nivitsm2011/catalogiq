"""Run every quality check on the whole TEST catalog and write the ops-dashboard data.

Needs the cached classifier probabilities from scripts/evaluate_consistency.py and the cached CLIP
embeddings from Phase 3. Outputs: reports/phase6/quality_report.csv (one row per listing) and
reports/phase6/quality_summary.json (issue rate, top issue types, by category, score distribution).
Usage: python scripts/run_quality_report.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from catalogiq.config import get_quality_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.quality.clip_checks import prompt_embeddings, text_margin
from catalogiq.quality.completeness import completeness
from catalogiq.quality.consistency import (
    clip_probabilities,
    flag_mismatch,
    load_consistency_thresholds,
    mismatch_signals,
    value_embeddings,
)
from catalogiq.quality.image_checks import compute_metrics, flags_from_metrics, load_thresholds
from catalogiq.quality.score import listing_score
from catalogiq.search.embeddings import ClipEncoder
from catalogiq.vision.predict import AttributePredictor

logger = get_logger("run_quality_report")
LISTING_FIELDS = ["gender", "articleType", "baseColour", "usage", "productDisplayName"]


def main() -> None:
    """Score every test listing and summarise."""
    cfg = get_quality_config()
    index = pd.read_parquet(resolve_path(cfg["paths"]["image_index"]))
    names = pd.read_parquet(resolve_path("data/processed/search_catalog.parquet"))[
        ["id", "productDisplayName"]
    ]
    index = index.merge(names, on="id", how="left")
    rows = np.where(index["split"].eq("test"))[0]
    table = index.iloc[rows].reset_index(drop=True)
    images = np.load(resolve_path(cfg["paths"]["image_cache"]), mmap_mode="r")
    clip_img = np.load(resolve_path("data/processed/clip_image_emb.npy"))[rows]
    clf = dict(np.load(resolve_path(cfg["paths"]["classifier_probs_dir"]) / "test.npz"))
    predictor = AttributePredictor()
    encoder = ClipEncoder()
    labels = predictor.labels
    value_emb = value_embeddings(encoder, labels)
    prompts = prompt_embeddings(encoder)
    margins = text_margin(clip_img, prompts)
    ith, cth = load_thresholds(), load_consistency_thresholds()
    attrs = cfg["consistency"]["attributes"]
    sigs = {}
    for a in attrs:
        to_idx = {v: i for i, v in enumerate(labels[a])}
        sigs[a] = mismatch_signals(
            table[a].map(to_idx).to_numpy(), clf[a], clip_probabilities(clip_img, value_emb[a])
        )
    flagged = {
        a: flag_mismatch(sigs[a], cth["min_conf"], cth["margin"], cth["mode"]) for a in attrs
    }
    records = []
    for i in range(len(table)):
        metrics = compute_metrics(np.asarray(images[rows[i]]))
        flags = flags_from_metrics(metrics, ith, float(margins[i]))
        raised = [k for k, v in flags.items() if v]
        mism = []
        for a in attrs:
            if flagged[a][i]:
                s = sigs[a]
                seen = (
                    labels[a][int(s["clf_top"].iat[i])]
                    if s["clf_alt_conf"].iat[i] >= cth["min_conf"]
                    else labels[a][int(s["clip_best_alt"].iat[i])]
                )
                mism.append({"attribute": a, "stated": str(table[a].iat[i]), "seen": seen})
        comp = completeness({f: table[f].iat[i] if f in table else None for f in LISTING_FIELDS})
        res = listing_score(raised, mism, comp["pct"], comp["missing"])
        records.append(
            {
                "id": int(table["id"].iat[i]),
                "masterCategory": table["masterCategory"].iat[i],
                "articleType": table["articleType"].iat[i],
                "group": comp["group"],
                "score": res["score"],
                "needs_attention": res["needs_attention"],
                "image_issues": ";".join(raised),
                "mismatches": ";".join(m["attribute"] for m in mism),
                "missing": ";".join(comp["missing"]),
                "completeness_pct": comp["pct"],
                "image_score": res["components"]["image"],
                "consistency_score": res["components"]["consistency"],
                "fixes": " | ".join(f["fix"] for f in res["flags"]),
                **{f"metric_{k}": round(v, 3) for k, v in metrics.items()},
                "text_margin": round(float(margins[i]), 3),
            }
        )
        if (i + 1) % 2000 == 0:
            logger.info("%d / %d", i + 1, len(table))
    df = pd.DataFrame(records)
    df.to_csv(resolve_path(cfg["paths"]["batch_csv"]), index=False)

    def issue_list(col: str) -> pd.Series:
        return df[col].str.split(";").explode().replace("", np.nan).dropna()

    has_issue = (df["image_issues"] != "") | (df["mismatches"] != "") | (df["missing"] != "")
    types = pd.concat(
        [
            issue_list("image_issues"),
            issue_list("mismatches").map(lambda a: f"mismatch_{a}"),
            issue_list("missing").map(lambda a: f"missing_{a}"),
        ]
    )
    summary = {
        "n_listings": len(df),
        "pct_with_any_issue": float(has_issue.mean() * 100),
        "pct_needs_attention": float(df["needs_attention"].mean() * 100),
        "score_mean": float(df["score"].mean()),
        "score_percentiles": {
            str(p): float(np.percentile(df["score"], p)) for p in (5, 25, 50, 75, 95)
        },
        "issue_counts": types.value_counts().to_dict(),
        "by_master_category": df.assign(issue=has_issue)
        .groupby("masterCategory")
        .agg(
            n=("id", "size"),
            pct_with_issue=("issue", lambda s: 100 * s.mean()),
            mean_score=("score", "mean"),
        )
        .round(2)
        .to_dict("index"),
        "by_group": df.assign(issue=has_issue)
        .groupby("group")
        .agg(
            n=("id", "size"),
            pct_with_issue=("issue", lambda s: 100 * s.mean()),
            mean_score=("score", "mean"),
        )
        .round(2)
        .to_dict("index"),
        "thresholds": {"image": ith, "consistency": cth},
    }
    resolve_path(cfg["paths"]["batch_summary"]).write_text(
        json.dumps(summary, indent=1, default=float)
    )
    logger.info(
        "issue rate %.1f%%, needs attention %.1f%%",
        summary["pct_with_any_issue"],
        summary["pct_needs_attention"],
    )


if __name__ == "__main__":
    main()
