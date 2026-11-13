"""Re-run the evaluation of each component (Phases 2-6) on its fixed test set.

Every function returns a ``ComponentResult`` with flat metric names
(``vision.articleType.macro_f1``,
``qa.correct_refusal_rate`` ...) that the scorecard, MLflow and the drift check all share.
Where an LLM is involved the replies come from the project's on-disk LLM cache; the number of calls
that missed the cache (and therefore really ran the model) is recorded in the provenance.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from catalogiq.config import (
    get_config,
    get_eval_config,
    get_qa_config,
    get_quality_config,
    get_rag_config,
    get_vision_config,
    resolve_path,
)
from catalogiq.eval.common import (
    ComponentResult,
    counted_cached_client,
    load_script,
    run_script,
)
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)
REPORTS = "reports"


# ---------------------------------------------------------------------------------------- vision
def run_vision() -> ComponentResult:
    """Phase 2: fresh inference of the exported classifier on the 6,339 test photos."""
    from catalogiq.quality.consistency import classifier_probabilities
    from catalogiq.vision import baselines as bl
    from catalogiq.vision.data import encode_labels
    from catalogiq.vision.evaluate import score_target
    from catalogiq.vision.predict import AttributePredictor, measure_latency

    start = time.perf_counter()
    vcfg = get_vision_config()
    index = pd.read_parquet(resolve_path(vcfg["image_index"]))
    images = np.load(resolve_path(vcfg["image_cache"]), mmap_mode="r")
    test = np.where(index["split"].eq("test"))[0]
    train = np.where(index["split"].eq("train"))[0]
    predictor = AttributePredictor()
    probs = classifier_probabilities(np.asarray(images[test]), predictor)
    maps = {t: predictor.labels[t] for t in predictor.targets}
    y_test = encode_labels(index.iloc[test].reset_index(drop=True), maps)
    y_train = encode_labels(index.iloc[train].reset_index(drop=True), maps)
    cached = np.load(resolve_path("data/interim/baseline_probs.npz"))
    thresholds = json.loads(
        (predictor.model_dir / vcfg["inference"]["thresholds_file"]).read_text(encoding="utf-8")
    )
    rows, metrics = [], {}
    for t in predictor.targets:
        y = y_test[t].numpy()
        sc = score_target(y, probs[t])
        maj = bl.majority_scores(y_train[t], y_test[t], len(maps[t]))
        logreg = score_target(y, cached[f"logreg_{t}"])
        clip = score_target(y, cached[f"clip_{t}"])
        valid = y >= 0
        conf, pred = probs[t].max(axis=1), probs[t].argmax(axis=1)
        accepted = valid & (conf >= thresholds[t]["threshold"])
        coverage = float(accepted.sum() / valid.sum())
        precision = (
            float((pred[accepted] == y[accepted]).mean()) if accepted.any() else float("nan")
        )
        rows.append(
            {
                "attribute": t,
                "macro_f1": sc["macro_f1"],
                "majority": maj["macro_f1"],
                "logreg": logreg["macro_f1"],
                "clip_zero_shot": clip["macro_f1"],
                "accuracy": sc["accuracy"],
                "top3": sc["top3_accuracy"],
                "threshold": thresholds[t]["threshold"],
                "coverage": coverage,
                "precision_accepted": precision,
            }
        )
        for key, val in {
            "macro_f1": sc["macro_f1"],
            "accuracy": sc["accuracy"],
            "coverage": coverage,
            "precision_accepted": precision,
        }.items():
            metrics[f"vision.{t}.{key}"] = val
    table = pd.DataFrame(rows)
    for col, key in (
        ("macro_f1", "vision.mean.macro_f1"),
        ("majority", "vision.majority.mean.macro_f1"),
        ("logreg", "vision.logreg.mean.macro_f1"),
        ("clip_zero_shot", "vision.clip_zero_shot.mean.macro_f1"),
    ):
        metrics[key] = float(table[col].mean())
    latency = measure_latency(predictor, np.asarray(images[test[0]]))
    metrics["vision.latency_mean_ms"] = latency["mean_ms"]
    metrics["vision.latency_p95_ms"] = latency["p95_ms"]
    # the quality checker reuses these probabilities from cache; confirm the cache is the same
    cache_path = resolve_path(get_quality_config()["paths"]["classifier_probs_dir"]) / "test.npz"
    drift_vs_cache = float("nan")
    if cache_path.exists():
        old = np.load(cache_path)
        drift_vs_cache = max(float(np.abs(old[t] - probs[t]).max()) for t in predictor.targets)
    return ComponentResult(
        "vision",
        metrics,
        {"per_attribute": table, "latency": latency},
        {
            "recomputed": "classifier inference on all test photos, scoring, coverage, latency",
            "cached_inputs": "baseline predictions (majority recomputed; logistic regression and "
            "CLIP zero-shot predictions cached from Phase 2 and re-scored)",
            "test_photos": int(len(test)),
            "max_abs_prob_diff_vs_phase6_cache": drift_vs_cache,
        },
        time.perf_counter() - start,
    )


# ---------------------------------------------------------------------------------------- search
def text_row_key(method: str) -> str:
    """Metric key of a row of ``text_search_metrics.csv`` (the file has three kinds of rows)."""
    if method == "CLIP text -> image embeddings":
        return "text"
    return "text_random" if method.startswith("Random") else "text_reference"


def run_search() -> ComponentResult:
    """Phase 3: image and text search metrics, index latency (``scripts/evaluate_search.py``)."""
    seconds = run_script("evaluate_search")
    img = pd.read_csv(resolve_path("reports/search_metrics.csv"))
    txt = pd.read_csv(resolve_path("reports/text_search_metrics.csv"))
    lat = json.loads(resolve_path("reports/search_latency.json").read_text(encoding="utf-8"))
    names = {
        "CLIP ViT-B/32 image embeddings": "clip",
        "Phase 2 fine-tuned classifier features": "clf_features",
        "Frozen ImageNet MobileNetV3 features": "frozen",
        "Random vectors (floor)": "random",
    }
    metrics: dict[str, float] = {}
    for r in img.to_dict("records"):
        key = names[r["representation"]]
        for col, name in (
            ("recall@5", "recall@5"),
            ("recall@10", "recall@10"),
            ("precision@10", "precision@10"),
            ("mAP@10", "map@10"),
        ):
            metrics[f"search.{key}.{name}"] = float(r[col])
    for r in txt.to_dict("records"):
        key = text_row_key(r["method"])
        metrics[f"search.{key}.recall@10"] = float(r["recall@10"])
        metrics[f"search.{key}.precision@10"] = float(r["precision@10"])
        metrics[f"search.{key}.map@10"] = float(r["mAP@10"])
    metrics["search.vector_search_ms"] = lat["vector_search_flat_unfiltered"]["mean_ms"]
    metrics["search.vector_search_filtered_ms"] = lat["vector_search_flat_filtered_gender_colour"][
        "mean_ms"
    ]
    metrics["search.clip_encode_image_ms"] = lat["clip_encode_one_image"]["mean_ms"]
    metrics["search.clip_encode_text_ms"] = lat["clip_encode_one_text"]["mean_ms"]
    return ComponentResult(
        "search",
        metrics,
        {"image": img, "text": txt, "latency": lat},
        {
            "recomputed": "retrieval over cached CLIP embeddings, metrics, latency",
            "cached_inputs": "CLIP embeddings of the 42,257 photos (about 2 h of CPU to recompute)",
        },
        seconds,
    )


def run_duplicates() -> ComponentResult:
    """Phase 3: duplicate threshold, precision and recall (``scripts/tune_duplicates.py``)."""
    labels = get_eval_config()["duplicates"]["labels_csv"]
    seconds = run_script("tune_duplicates", ["--labels", labels])
    res = json.loads(resolve_path("reports/duplicate_threshold.json").read_text(encoding="utf-8"))
    stats = json.loads(
        resolve_path("reports/duplicate_label_stats.json").read_text(encoding="utf-8")
    )
    metrics = {
        "duplicates.threshold": res["recommended_threshold"],
        "duplicates.precision": res["recommended_precision"],
        "duplicates.recall": res["recommended_recall"],
        "duplicates.f1_optimal_precision": res["f1_optimal_precision"],
        "duplicates.f1_optimal_recall": res["f1_optimal_recall"],
    }
    return ComponentResult(
        "duplicates",
        metrics,
        {"result": res, "label_stats": stats},
        {
            "recomputed": "threshold rule on the 100 labelled pairs",
            "labels": "PROVISIONAL: assigned by the AI assistant, not by a person",
        },
        seconds,
    )


# --------------------------------------------------------------------------------------- listing
def run_listing() -> ComponentResult:
    """Phase 4: regenerate and re-score 50 products x 4 arms through the LLM cache."""
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

    start = time.perf_counter()
    out = resolve_path(get_eval_config()["paths"]["rag_regen_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("gen_*.jsonl"):
        old.unlink()
    saved = pd.read_parquet(
        resolve_path(get_rag_config()["paths"]["eval_dir"]) / "products.parquet"
    )
    products = sample_products()
    attributes = predict_attributes(products)
    llm, counter = counted_cached_client()
    lexicon = load_brand_lexicon()
    retriever = HybridRetriever()
    gen = ListingGenerator(llm, retriever, lexicon)
    metrics, rows = {}, []
    for use_rag, fix in ((False, False), (True, False), (False, True), (True, True)):
        mode = arm_name(use_rag, fix)
        rec = run_generation(gen, products, attributes, use_rag, out / f"gen_{mode}.jsonl", fix)
        rec = rescore_reports(rec, lexicon)
        judged = run_judge(gen, rec)
        s = summarise_run(rec)
        s["judge_clarity"] = float(judged["judge_clarity"].mean())
        s["judge_accuracy"] = float(judged["judge_accuracy"].mean())
        if use_rag:
            s["retrieval_hit_rate"] = float(retrieval_hit(rec, retriever.items).mean())
        for k, v in s.items():
            metrics[f"listing.{mode}.{k}"] = float(v)
        rows.append({"arm": mode, **s})
    return ComponentResult(
        "listing",
        metrics,
        {"arms": pd.DataFrame(rows)},
        {
            "recomputed": "product sample, classifier attributes, retrieval, guardrails, judge "
            "scoring",
            "llm": "replies read from the LLM cache (a cached reply keeps its original latency)",
            "llm_calls_that_missed_cache": counter.calls,
            "same_products_as_phase4": bool(set(saved["id"]) == set(products["id"])),
            "n_products": int(len(products)),
        },
        time.perf_counter() - start,
    )


# --------------------------------------------------------------------------------------------- qa
def run_qa() -> ComponentResult:
    """Phase 5: retrieval, answers, faithfulness judge and refusals for the 60 questions."""
    from catalogiq.rag.eval_qa import (
        answer_metrics,
        ground_truth,
        load_questions,
        review_texts,
        validate_questions,
    )
    from catalogiq.rag.knowledge_base import get_client
    from catalogiq.rag.qa import ReviewQA
    from catalogiq.rag.reviews_index import load_chunks

    start = time.perf_counter()
    ev = load_script("evaluate_qa")
    qcfg = get_qa_config()
    out = resolve_path(get_eval_config()["paths"]["qa_regen_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for old in list(out.glob("answers.jsonl")) + list(out.glob("faithfulness.jsonl")):
        old.unlink()
    reviews = review_texts()
    gt = ground_truth(load_questions(), reviews)
    assert not validate_questions(gt), "test set no longer valid"
    chunks = load_chunks()
    col = get_client().get_collection(qcfg["paths"]["collection"])
    meta = pd.read_parquet(resolve_path(get_config()["data"]["reviews"]["meta_parquet"]))
    titles = dict(zip(meta["parent_asin"], meta["title"], strict=True))
    retrieval = ev.stage_retrieval(ReviewQA(None, chunks, col, titles), gt, out)  # type: ignore[arg-type]
    gate = json.loads((out / "gate.json").read_text(encoding="utf-8"))
    llm, counter = counted_cached_client()
    qa = ReviewQA(llm, chunks, col, titles, gate["min_similarity"])
    ev.stage_generate(qa, gt, out, reviews)
    ev.stage_judge(llm, out, reviews)
    answers = pd.DataFrame(
        [json.loads(x) for x in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    )
    faith = pd.DataFrame(
        [
            json.loads(x)
            for x in (out / "faithfulness.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    )
    text_of = dict(zip(reviews["review_id"], reviews["full"], strict=True))
    m = answer_metrics(answers, faith, gt.set_index("id", drop=False), text_of)
    metrics = {f"qa.{k}": float(v) for k, v in m.items() if "latency" not in k}
    # a cached reply replays instantly, so the regenerated latency is not a measurement; take the
    # real model time from the original Phase 5 run (the system test re-measures it uncached)
    original = pd.DataFrame(
        [
            json.loads(x)
            for x in (resolve_path("reports/phase5/answers.jsonl"))
            .read_text(encoding="utf-8")
            .splitlines()
        ]
    )
    refused = original["output"].map(lambda o: o["refused"])
    metrics["qa.latency_answered_s"] = float(original[~refused]["latency_s"].mean())
    metrics["qa.latency_refused_s"] = float(original[refused]["latency_s"].mean())
    ans = retrieval[retrieval["kind"] == "answerable"]
    variants = {}
    for variant, g in ans.groupby("variant"):
        variants[variant] = {
            "recall5": float(g["recall5"].mean()),
            "mrr": float(g["mrr"].mean()),
            "latency_ms": float(retrieval[retrieval["variant"] == variant]["latency_ms"].mean()),
        }
        metrics[f"qa.retrieval.{variant}.recall5"] = variants[variant]["recall5"]
        metrics[f"qa.retrieval.{variant}.mrr"] = variants[variant]["mrr"]
    metrics["qa.gate_threshold"] = float(gate["min_similarity"])
    metrics["qa.gate_cv_answer_rate"] = float(gate["cross_validated"]["answer_rate"])
    metrics["qa.gate_cv_refuse_rate"] = float(gate["cross_validated"]["refuse_rate"])
    # owner hand-check (a person; not re-runnable) against the freshly regenerated judge
    key = pd.read_csv(resolve_path("reports/phase5/qa_handcheck_key.csv"))
    hum = pd.read_csv(resolve_path("reports/phase5/qa_handcheck_ratings.csv"))
    judged = {r["id"]: bool(r["judge"]["faithful"]) for r in faith.to_dict("records")}
    d = key.merge(hum, on="item")
    d["human"] = d["faithful"].eq("yes")
    d["judge_now"] = d["id"].map(judged)
    metrics["qa.handcheck_agreement"] = float((d["judge_now"] == d["human"]).mean())
    metrics["qa.handcheck_human_faithful"] = float(d["human"].mean())
    return ComponentResult(
        "qa",
        metrics,
        {"retrieval": variants, "gate": gate, "end_to_end": m},
        {
            "recomputed": "retrieval variants, gate tuning, answers, judge, citation checks",
            "llm": "replies read from the LLM cache",
            "latency": "taken from the original Phase 5 run (cached replies replay instantly)",
            "llm_calls_that_missed_cache": counter.calls,
            "human_input": "faithfulness hand-check of 15 answers by the owner (saved ratings)",
        },
        time.perf_counter() - start,
    )


# ----------------------------------------------------------------------------------------- quality
def run_quality(reuse_image_eval: bool = False) -> ComponentResult:
    """Phase 6: image checks, consistency with injected mismatches, batch report."""
    from catalogiq.quality.evaluate import per_check_scores, recall_by_level

    start = time.perf_counter()
    qcfg = get_quality_config()
    out = resolve_path(qcfg["paths"]["eval_dir"])
    seconds = {}
    if not reuse_image_eval:
        seconds["evaluate_image_checks"] = run_script("evaluate_image_checks")
    seconds["evaluate_consistency"] = run_script("evaluate_consistency")
    seconds["run_quality_report"] = run_script("run_quality_report")
    flags = pd.read_csv(out / "image_eval_flags.csv")
    scores = per_check_scores(flags)
    metrics: dict[str, float] = {}
    for check, row in scores.iterrows():
        metrics[f"quality.{check}.recall"] = float(row["recall"])
        metrics[f"quality.{check}.precision"] = float(row["precision"])
        metrics[f"quality.{check}.clean_flag_rate"] = float(row["false_alarm_rate_clean"])
    cons = json.loads((out / "consistency_eval.json").read_text(encoding="utf-8"))
    for attr, row in cons["test"].items():
        for k in ("precision", "recall", "f1"):
            metrics[f"consistency.{attr}.{k}"] = float(row[k])
    summary = json.loads(resolve_path(qcfg["paths"]["batch_summary"]).read_text(encoding="utf-8"))
    for k in ("pct_with_any_issue", "pct_needs_attention", "score_mean"):
        metrics[f"quality.batch.{k}"] = float(summary[k])
    return ComponentResult(
        "quality",
        metrics,
        {
            "checks": scores,
            "by_level": recall_by_level(flags),
            "consistency": cons,
            "batch": summary,
        },
        {
            "recomputed": "degraded copies and flags, consistency injection and scoring "
            "(thresholds "
            "kept as tuned), batch report over all test listings",
            "reused_image_eval": reuse_image_eval,
            "labels": "natural-photo labels and the 24-listing review are the AI assistant's",
            "seconds_by_script": seconds,
        },
        time.perf_counter() - start,
    )


RUNNERS = {
    "vision": run_vision,
    "search": run_search,
    "duplicates": run_duplicates,
    "listing": run_listing,
    "qa": run_qa,
    "quality": run_quality,
}
