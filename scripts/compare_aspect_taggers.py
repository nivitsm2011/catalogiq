"""Phase 5 step 2: compare aspect taggers on the 200 hand-labelled review chunks.

Methods: keyword rules, MiniLM embedding similarity, NLI zero-shot (DeBERTa-v3-xsmall) and the
local LLM as a zero-shot classifier. Thresholds of the two score-based methods are tuned on a "dev"
half of the labelled sample; every method is reported on the untouched "test" half (and on all 200).
Selection rule (fixed before running): highest macro-F1 on the test half; if the best is within
0.03 of the keyword rules, keep the rules (cheaper and explainable).
Usage: python scripts/compare_aspect_taggers.py [--no-llm]
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.llm.client import get_llm_client
from catalogiq.logging_utils import get_logger
from catalogiq.rag.aspects import (
    aspect_names,
    best_threshold,
    embedding_scores,
    llm_tag,
    nli_scores,
    prf,
    rule_tag,
    timed,
)

logger = get_logger("compare_aspect_taggers")
SELECTION_MARGIN = 0.03


def main() -> None:
    cfg = get_qa_config()
    zs = cfg["zero_shot"]
    names = aspect_names()
    sample = pd.read_csv(resolve_path(cfg["paths"]["aspect_sample"]))
    labels = pd.read_csv(resolve_path(cfg["paths"]["aspect_labels"]))
    df = sample.merge(labels, on="chunk_id", how="inner")
    logger.info("%d labelled chunks (of %d sampled)", len(df), len(sample))
    y = df[names].to_numpy().astype(bool)
    texts = df["text"].tolist()
    rng = np.random.default_rng(cfg["seed"])
    order = rng.permutation(len(df))
    dev, test = order[: len(df) // 2], order[len(df) // 2 :]

    scores: dict[str, tuple[np.ndarray, float]] = {}
    rules, t_rules = timed(lambda: rule_tag(texts))
    scores["rules"] = (rules, t_rules)
    emb, t_emb = timed(lambda: embedding_scores(texts))
    nli, t_nli = timed(lambda: nli_scores(texts))
    grid = np.round(np.arange(0.05, 0.96, 0.025), 3)
    thr = {
        "embedding": best_threshold(emb[dev], y[dev], grid),
        "nli": best_threshold(nli[dev], y[dev], grid),
    }
    scores["embedding"] = (emb >= thr["embedding"], t_emb)
    scores["nli"] = (nli >= thr["nli"], t_nli)
    if "--no-llm" not in sys.argv:
        llm_pred, t_llm = timed(lambda: llm_tag(texts, get_llm_client(cache=False)))
        scores["llm"] = (llm_pred, t_llm)

    summary: dict[str, dict] = {}
    per_aspect: dict[str, pd.DataFrame] = {}
    for method, (pred, secs) in scores.items():
        res_test = prf(y[test], pred[test]).set_index("aspect")
        res_all = prf(y, pred).set_index("aspect")
        per_aspect[method] = res_test
        summary[method] = {
            "macro_f1_test": float(res_test.loc["MACRO", "f1"]),
            "micro_f1_test": float(res_test.loc["MICRO", "f1"]),
            "macro_f1_all": float(res_all.loc["MACRO", "f1"]),
            "micro_f1_all": float(res_all.loc["MICRO", "f1"]),
            "macro_precision_test": float(res_test.loc["MACRO", "precision"]),
            "macro_recall_test": float(res_test.loc["MACRO", "recall"]),
            "seconds_per_1000_chunks": 1000 * secs / len(df),
        }
    best = max(summary, key=lambda m: summary[m]["macro_f1_test"])
    chosen = (
        "rules"
        if summary[best]["macro_f1_test"] - summary["rules"]["macro_f1_test"] <= SELECTION_MARGIN
        else best
    )
    out = {
        "n_labelled": len(df),
        "prevalence": dict(zip(names, y.sum(axis=0).astype(int).tolist(), strict=True)),
        "thresholds": thr,
        "summary": summary,
        "best_by_macro_f1_test": best,
        "chosen": chosen,
        "per_aspect_test_f1": {m: r["f1"].round(3).to_dict() for m, r in per_aspect.items()},
    }
    path = resolve_path(cfg["paths"]["aspect_results"])
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    logger.info("\n%s", pd.DataFrame(summary).T.round(3).to_string())
    logger.info("best=%s chosen=%s thresholds=%s", best, chosen, thr)
    _ = zs


if __name__ == "__main__":
    main()
