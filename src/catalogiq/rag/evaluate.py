"""Phase 4 evaluation helpers: held-out sample, seller notes, resumable runs, summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_rag_config, resolve_path
from catalogiq.data.fashion import image_path
from catalogiq.logging_utils import get_logger
from catalogiq.rag.categories import group_from_article_type
from catalogiq.rag.guardrails import Attributes
from catalogiq.rag.judge import judge_listing
from catalogiq.rag.listing import ListingGenerator

logger = get_logger(__name__)


def load_brand_lexicon() -> list[str]:
    """Brand names written by ``scripts/build_knowledge_base.py`` (empty list if missing)."""
    path = resolve_path(get_rag_config()["paths"]["brand_lexicon"])
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def extract_brand(product_name: str, lexicon: set[str]) -> str | None:
    """Longest 1-3 word prefix of the catalog product name that is a known brand."""
    words = str(product_name).split()
    for n in (3, 2, 1):
        cand = " ".join(words[:n])
        if len(words) > n and cand.lower() in lexicon:
            return cand
    return None


def sample_products(n: int | None = None, seed: int | None = None) -> pd.DataFrame:
    """Stratified (by category group) sample of held-out products from the Kaggle test split."""
    cfg = get_rag_config()
    n = n or cfg["evaluation"]["n_products"]
    seed = cfg["seed"] if seed is None else seed
    test = pd.read_parquet(
        resolve_path(f"data/processed/fashion_{cfg['evaluation']['split']}.parquet")
    )
    test = test[test["image_ok"]].copy()
    test["group"] = test["articleType"].map(group_from_article_type)
    test = test[test["group"] != cfg["categories"]["default_group"]]
    share = test["group"].value_counts(normalize=True)
    raw = share * n
    quota = np.floor(raw).astype(int).clip(lower=1)
    while quota.sum() < n:  # largest remainder
        quota[(raw - quota).idxmax()] += 1
    while quota.sum() > n:
        quota[quota.idxmax()] -= 1
    parts = [
        test[test["group"] == g].sample(int(q), random_state=seed)
        for g, q in quota.items()
        if q > 0
    ]
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    lex = {b.lower() for b in load_brand_lexicon()}
    out["brand"] = out["productDisplayName"].map(lambda s: extract_brand(s, lex))
    return out


def seller_notes_for(row: pd.Series) -> dict[str, Any]:
    """Seller notes simulated from the catalog name: brand only (Kaggle has no materials)."""
    if get_rag_config()["evaluation"]["seller_note_brand"] and row.get("brand"):
        return {"brand": row["brand"]}
    return {}


def predict_attributes(products: pd.DataFrame) -> dict[int, Attributes]:
    """Phase 2 classifier predictions for every sampled product photo (image -> attributes)."""
    from catalogiq.vision.predict import predict

    images_dir = get_config()["data"]["fashion"]["images_dir"]
    out: dict[int, Attributes] = {}
    for pid in products["id"]:
        out[int(pid)] = predict(image_path(resolve_path(images_dir), int(pid)))
    return out


def arm_name(use_rag: bool, self_correct: bool) -> str:
    """Name of an evaluation arm: no_rag, rag, no_rag_fix or rag_fix."""
    return ("rag" if use_rag else "no_rag") + ("_fix" if self_correct else "")


def run_generation(
    gen: ListingGenerator,
    products: pd.DataFrame,
    attributes: dict[int, Attributes],
    use_rag: bool,
    out_path: str | Path,
    self_correct: bool = False,
) -> pd.DataFrame:
    """Generate (resumable: products already in ``out_path`` are skipped); return all records."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[int, dict[str, Any]] = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            done[rec["id"]] = rec
    with out_path.open("a", encoding="utf-8") as fh:
        for _, row in products.iterrows():
            pid = int(row["id"])
            if pid in done:
                continue
            attrs, notes = attributes[pid], seller_notes_for(row)
            res = gen.generate(attrs, notes, use_rag=use_rag, self_correct=self_correct)
            rec = {
                "id": pid,
                "mode": arm_name(use_rag, self_correct),
                "group": res.group,
                "attributes": {k: list(v) for k, v in attrs.items()},
                "notes": notes,
                "ok": res.ok,
                "error": res.error,
                "attempts": res.attempts,
                "fixed": res.fixed,
                "fix_attempts": res.fix_attempts,
                "latency_s": res.latency_s,
                "retrieved": res.retrieved,
                "listing": res.listing.model_dump() if res.listing else None,
                "report": res.report.to_dict() if res.report else None,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[pid] = rec
            logger.info(
                "%s %d/%d id=%d ok=%s %.1fs",
                rec["mode"],
                len(done),
                len(products),
                pid,
                res.ok,
                res.latency_s,
            )
    return pd.DataFrame([done[int(i)] for i in products["id"]])


def retrieval_hit(records: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """Per record: does any retrieved example belong to the product's category group?"""
    grp = dict(zip(items["parent_asin"], items["group"], strict=True))
    return records.apply(
        lambda r: (
            float(any(grp.get(a) == r["group"] for a in r["retrieved"]))
            if r["retrieved"]
            else np.nan
        ),
        axis=1,
    )


def summarise_run(records: pd.DataFrame) -> dict[str, float]:
    """Headline metrics of one run (one mode)."""
    ok = records[records["ok"]]
    rep = pd.DataFrame(list(ok["report"]))
    return {
        "n": len(records),
        "valid_json_rate": float(records["ok"].mean()),
        "repair_used_rate": float((records["attempts"] > 1).mean()),
        "attribute_coverage": float(rep["coverage"].mean()),
        "hallucination_rate": float(rep["hallucinated"].mean()),
        "unsupported_per_listing": float(rep["unsupported"].map(len).mean()),
        "style_compliance": float(rep["style_compliant"].mean()),
        "banned_rate": float(rep["banned"].map(bool).mean()),
        "latency_mean_s": float(records["latency_s"].mean()),
        "latency_median_s": float(records["latency_s"].median()),
    }


def run_judge(
    gen: ListingGenerator,
    records: pd.DataFrame,
) -> pd.DataFrame:
    """Add judge clarity/accuracy columns to the records (``NaN`` where the judge failed)."""
    from catalogiq.rag.schema import Listing

    clar, acc, why = [], [], []
    for r in records.itertuples():
        if not r.ok:
            clar.append(np.nan), acc.append(np.nan), why.append("")
            continue
        attrs = {k: (v[0], v[1], v[2]) for k, v in r.attributes.items()}
        score = judge_listing(gen.llm, Listing.model_validate(r.listing), attrs, r.notes)
        clar.append(score.clarity if score else np.nan)
        acc.append(score.accuracy if score else np.nan)
        why.append(score.reason if score else "")
    out = records.copy()
    out["judge_clarity"], out["judge_accuracy"], out["judge_reason"] = clar, acc, why
    return out


def rescore_reports(records: pd.DataFrame, brand_lexicon: list[str]) -> pd.DataFrame:
    """Re-run the (current) guardrails on stored listings so every arm uses identical checks."""
    from catalogiq.rag.categories import group_spec
    from catalogiq.rag.guardrails import check_listing
    from catalogiq.rag.schema import Listing

    out = records.copy()
    reports = []
    for r in records.itertuples():
        if not r.ok:
            reports.append(None)
            continue
        attrs = {k: (v[0], v[1], v[2]) for k, v in r.attributes.items()}
        rep = check_listing(
            Listing.model_validate(r.listing),
            attrs,
            r.notes,
            group_spec(r.group)["required_attributes"],
            brand_lexicon,
        )
        reports.append(rep.to_dict())
    out["report"] = reports
    return out
