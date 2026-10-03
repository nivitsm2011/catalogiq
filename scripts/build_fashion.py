"""Audit, clean and split the Fashion Product Images dataset.

Run after the Kaggle download:  python scripts/build_fashion.py
Outputs: data/interim/fashion_audit.parquet, reports/fashion_audit.json,
data/processed/fashion_{clean,train,val,test}.parquet, reports/fashion_cleaning_log.json,
reports/target_predictability.csv
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from catalogiq.config import get_config, resolve_path
from catalogiq.data.audit import audit_summary, inspect_images, near_duplicate_groups
from catalogiq.data.cleaning import clean_fashion, load_colour_map
from catalogiq.data.fashion import load_styles
from catalogiq.data.splits import make_splits, verify_no_leakage
from catalogiq.data.target_evidence import target_evidence
from catalogiq.logging_utils import get_logger
from catalogiq.utils import set_seed, timed

logger = get_logger(__name__)


@timed
def main() -> None:
    """Run the fashion data pipeline end to end."""
    cfg = get_config()
    set_seed(cfg["project"]["seed"])
    fcfg, ccfg = cfg["data"]["fashion"], cfg["fashion_cleaning"]
    taxonomy = fcfg["taxonomy"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reuse-audit", action="store_true", help="skip the image scan; reuse the audit parquet"
    )
    args = parser.parse_args()

    df, report = load_styles(resolve_path(fcfg["styles_csv"]))
    audit_file = resolve_path(fcfg["audit_parquet"])
    if args.reuse_audit and audit_file.exists():
        cached = pd.read_parquet(audit_file)
        df = df.merge(
            cached[["id", "image_exists", "image_ok", "phash", "width", "height"]], on="id"
        )
    else:
        df = inspect_images(df, resolve_path(fcfg["images_dir"]), ccfg["min_image_side"])
    df["dup_group"] = near_duplicate_groups(df["phash"], ccfg["phash_hamming_threshold"])
    # unreadable images have no hash; give each its own group so they never pool together
    no_hash = df["dup_group"] < 0
    df.loc[no_hash, "dup_group"] = df["dup_group"].max() + 1 + range(int(no_hash.sum()))

    audit_path = resolve_path(fcfg["audit_parquet"])
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(audit_path, index=False)
    summary = audit_summary(df.assign(dup_group=df["dup_group"]), taxonomy, report.as_dict())
    out_json = resolve_path(fcfg["audit_summary_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    clean, log = clean_fashion(
        df, load_colour_map(resolve_path(fcfg["colour_map"])), ccfg["rare"], taxonomy
    )
    clean = make_splits(
        clean,
        stratify_col="articleType",
        group_col="dup_group",
        fractions=ccfg["split_fractions"],
        n_folds=ccfg["split_folds"],
        seed=cfg["project"]["seed"],
    )
    assert verify_no_leakage(clean, "dup_group") == 0, "near-duplicate leakage across splits"

    out_dir = resolve_path(fcfg["splits_dir"])
    clean.to_parquet(resolve_path(fcfg["clean_parquet"]), index=False)
    for name in ("train", "val", "test"):
        part = clean[clean["split"] == name]
        part.to_parquet(out_dir / f"fashion_{name}.parquet", index=False)
        logger.info("split %-5s %6d rows (%.1f%%)", name, len(part), 100 * len(part) / len(clean))

    evidence = target_evidence(df, clean, taxonomy, "dup_group")
    evidence.to_csv(resolve_path("reports/target_evidence.csv"), index=False)
    resolve_path("reports/fashion_cleaning_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
