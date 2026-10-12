"""Phase 4 evaluation: 50 held-out products, no-RAG vs RAG, guardrail metrics and LLM-as-judge.

Resumable: generation appends to jsonl files, the LLM replies are cached on disk and judged
results are saved per mode. Usage: python scripts/evaluate_rag.py [--no-judge]
"""

from __future__ import annotations

import json
import sys

import pandas as pd

from catalogiq.config import get_rag_config, resolve_path
from catalogiq.llm.client import get_llm_client
from catalogiq.logging_utils import get_logger
from catalogiq.rag.evaluate import (
    arm_name,
    load_brand_lexicon,
    predict_attributes,
    rescore_reports,
    retrieval_hit,
    run_generation,
    run_judge,
    sample_products,
    summarise_run,
)
from catalogiq.rag.listing import ListingGenerator
from catalogiq.rag.retrieval import HybridRetriever
from catalogiq.tracking import start_run
from catalogiq.utils import set_seed

logger = get_logger("evaluate_rag")


def main() -> None:
    cfg = get_rag_config()
    set_seed(cfg["seed"])
    out = resolve_path(cfg["paths"]["eval_dir"])
    out.mkdir(parents=True, exist_ok=True)

    products = sample_products()
    products.to_parquet(out / "products.parquet")
    attrs_path = out / "attributes.json"
    attributes = predict_attributes(products)
    attrs = {k: {a: list(v) for a, v in d.items()} for k, d in attributes.items()}
    attrs_path.write_text(json.dumps(attrs))
    groups = products["group"].value_counts().to_dict()
    logger.info("Sampled %d products; groups: %s", len(products), groups)

    retriever = HybridRetriever()
    gen = ListingGenerator(get_llm_client(), retriever, load_brand_lexicon())
    runs = {}
    for use_rag, fix in ((False, False), (True, False), (False, True), (True, True)):
        mode = arm_name(use_rag, fix)
        runs[mode] = run_generation(
            gen, products, attributes, use_rag, out / f"gen_{mode}.jsonl", fix
        )
    if "--no-judge" in sys.argv:
        return
    summary = {}
    lexicon = load_brand_lexicon()
    for mode, rec in runs.items():
        rec = rescore_reports(rec, lexicon)
        judged_path = out / f"judged_{mode}.pkl"
        judged = pd.read_pickle(judged_path) if judged_path.exists() else run_judge(gen, rec)
        judged.to_pickle(judged_path)
        s = summarise_run(rec)
        s["judge_clarity"] = float(judged["judge_clarity"].mean())
        s["judge_accuracy"] = float(judged["judge_accuracy"].mean())
        if mode.startswith("rag"):
            s["retrieval_hit_rate"] = float(retrieval_hit(rec, retriever.items).mean())
        summary[mode] = s
        logger.info("%s: %s", mode, s)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    params = {"n_products": len(products), "model": gen.llm.model}
    with start_run("phase4_listing_generator", params):
        import mlflow

        for mode, s in summary.items():
            mlflow.log_metrics({f"{mode}_{k}": v for k, v in s.items()})


if __name__ == "__main__":
    main()
