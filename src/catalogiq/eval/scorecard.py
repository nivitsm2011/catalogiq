# ruff: noqa: E501  (long report prose)
"""Render reports/scorecard.md from the saved evaluation results (no model is run here)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from catalogiq.config import get_eval_config, resolve_path
from catalogiq.eval.common import table_to_markdown

ATTR_ORDER = [
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "gender",
    "season",
    "usage",
]


def _df(records: list[dict[str, Any]] | None) -> pd.DataFrame:
    return pd.DataFrame(records or [])


def _pct(v: float) -> str:
    return "" if v != v else f"{100 * v:.1f}%"


def _vision(c: dict[str, Any]) -> str:
    t = _df(c["tables"]["per_attribute"]).set_index("attribute").loc[ATTR_ORDER].reset_index()
    m = c["metrics"]
    mean = {
        "attribute": "**mean**",
        "macro_f1": m["vision.mean.macro_f1"],
        "majority": m["vision.majority.mean.macro_f1"],
        "logreg": m["vision.logreg.mean.macro_f1"],
        "clip_zero_shot": m["vision.clip_zero_shot.mean.macro_f1"],
    }
    t = pd.concat([t, pd.DataFrame([mean])], ignore_index=True)
    out = t[
        [
            "attribute",
            "macro_f1",
            "majority",
            "logreg",
            "clip_zero_shot",
            "accuracy",
            "threshold",
            "coverage",
            "precision_accepted",
        ]
    ].copy()
    out.columns = [
        "Attribute",
        "Macro-F1 (ours)",
        "Majority class",
        "Frozen MobileNet + logreg",
        "Zero-shot CLIP",
        "Accuracy",
        "Confidence threshold",
        "Coverage at threshold",
        "Precision of accepted",
    ]
    out["Coverage at threshold"] = out["Coverage at threshold"].map(
        lambda v: _pct(v) if v == v else ""
    )
    lat = c["tables"]["latency"]
    return (
        "Test split: 6,339 photos. Coverage = share of photos whose calibrated confidence clears the attribute's "
        "threshold (chosen so that accepted predictions are at least about 90% right on validation).\n\n"
        + table_to_markdown(out)
        + f"\n\nLatency (one photo, CPU, preprocessing included): mean {lat['mean_ms']:.1f} ms, "
        f"median {lat['median_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms over {lat['runs']} runs."
    )


def _search(c: dict[str, Any], dup: dict[str, Any]) -> str:
    img = _df(c["tables"]["image"])[
        ["representation", "queries", "recall@5", "recall@10", "precision@10", "mAP@10"]
    ]
    img.columns = ["Representation", "Queries", "Recall@5", "Recall@10", "Precision@10", "mAP@10"]
    txt = _df(c["tables"]["text"])[["method", "queries", "recall@10", "precision@10", "mAP@10"]]
    txt.columns = ["Text-to-image", "Queries", "Recall@10", "Precision@10", "mAP@10"]
    lat, d = c["tables"]["latency"], dup["metrics"]
    stats = dup["tables"]["label_stats"]
    return (
        "Image queries: 1,995 test photos against a 35,918-photo gallery; a result is relevant if it has the "
        "same product type AND colour.\n\n"
        + table_to_markdown(img)
        + "\n\n"
        + table_to_markdown(txt)
        + f"\n\nLatency over the 42,257-photo catalog: vector search {lat['vector_search_flat_unfiltered']['mean_ms']:.1f} ms "
        f"({lat['vector_search_flat_filtered_gender_colour']['mean_ms']:.1f} ms filtered); CLIP encodes one photo in "
        f"{lat['clip_encode_one_image']['mean_ms']:.0f} ms and one text in {lat['clip_encode_one_text']['mean_ms']:.0f} ms.\n\n"
        "**Duplicate detection** (CLIP similarity and pHash; threshold chosen for at least 95% precision): "
        f"threshold {d['duplicates.threshold']:.3f}, precision {d['duplicates.precision']:.2f}, recall {d['duplicates.recall']:.2f}. "
        f"Labelled pairs: {stats['duplicate']} duplicate / {stats['not_duplicate']} not / {stats['unsure']} unsure. "
        "**The labels are provisional (assigned by the AI assistant), and recall rests on very few positives; treat it as unknown.**"
    )


def _listing(c: dict[str, Any]) -> str:
    arms = _df(c["tables"]["arms"]).set_index("arm")
    names = {
        "no_rag": "no-RAG",
        "rag": "RAG",
        "no_rag_fix": "no-RAG + rewrite",
        "rag_fix": "RAG + rewrite",
    }
    rows = []
    for arm, label in names.items():
        r = arms.loc[arm]
        rows.append(
            {
                "Arm": label,
                "Hallucination rate (listings with an unsupported fact)": _pct(
                    r["hallucination_rate"]
                ),
                "Attribute coverage": _pct(r["attribute_coverage"]),
                "Style compliance": _pct(r["style_compliance"]),
                "Banned-word rate": _pct(r["banned_rate"]),
                "Judge clarity / accuracy (1-5)": f"{r['judge_clarity']:.2f} / {r['judge_accuracy']:.2f}",
                "Seconds per listing": f"{r['latency_mean_s']:.1f}",
            }
        )
    rag, no = arms.loc["rag"], arms.loc["no_rag"]
    return (
        "50 held-out products, seller note = brand only. Hallucination = a listing states a material, claim, number or brand "
        "that is not in the attributes or seller note (deterministic check). The judge is the same 3B model that writes the "
        "listings and agreed poorly with the owner on 10 listings, so it is a secondary metric.\n\n"
        + table_to_markdown(pd.DataFrame(rows))
        + f"\n\n**RAG vs no-RAG:** hallucination {_pct(rag['hallucination_rate'])} vs {_pct(no['hallucination_rate'])}, "
        f"style compliance {_pct(rag['style_compliance'])} vs {_pct(no['style_compliance'])}, judge accuracy "
        f"{rag['judge_accuracy']:.2f} vs {no['judge_accuracy']:.2f}, judge clarity {rag['judge_clarity']:.2f} vs "
        f"{no['judge_clarity']:.2f}, time {rag['latency_mean_s']:.0f} s vs {no['latency_mean_s']:.0f} s. "
        f"Retrieval returned the right category for {_pct(rag['retrieval_hit_rate'])} of products. "
        "Latencies are the original generation times replayed from the LLM cache."
    )


def _qa(c: dict[str, Any]) -> str:
    m = c["metrics"]
    ret = pd.DataFrame(c["tables"]["retrieval"]).T.loc[
        ["bm25", "vector", "hybrid", "hybrid_rerank"]
    ]
    ret = ret.reset_index()
    ret.columns = ["Retrieval variant", "Recall@5", "MRR", "Search ms"]
    ret["Retrieval variant"] = ret["Retrieval variant"].map(
        {
            "bm25": "BM25",
            "vector": "Vector",
            "hybrid": "Hybrid (RRF)",
            "hybrid_rerank": "Hybrid + helpfulness re-rank (used)",
        }
    )
    head = pd.DataFrame(
        [
            {
                "Metric": "Correct-refusal rate (15 unanswerable questions)",
                "Value": _pct(m["qa.correct_refusal_rate"]),
            },
            {
                "Metric": "False-refusal rate (45 answerable questions)",
                "Value": _pct(m["qa.false_refusal_rate"]),
            },
            {
                "Metric": "Citation validity (cited ids that exist among the shown reviews)",
                "Value": _pct(m["qa.citation_validity"]),
            },
            {
                "Metric": "Citation accuracy (cited review contains the answer pattern)",
                "Value": _pct(m["qa.citation_relevance"]),
            },
            {
                "Metric": "Faithful answers, LLM judge (a LOWER BOUND: the judge is too strict)",
                "Value": _pct(m["qa.faithful_answers"]),
            },
            {
                "Metric": "Faithful answers, owner hand-check of 15 answers (small sample)",
                "Value": _pct(m["qa.handcheck_human_faithful"]),
            },
            {
                "Metric": "Judge vs owner agreement on those 15",
                "Value": _pct(m["qa.handcheck_agreement"]),
            },
            {
                "Metric": "Latency per answered question / per refusal (CPU, original Phase 5 run)",
                "Value": f"{m['qa.latency_answered_s']:.1f} s / {m['qa.latency_refused_s']:.1f} s",
            },
        ]
    )
    return (
        "60 questions on 20 products (45 answerable, 15 unanswerable). Recall@5 = an answer-bearing review is among the top 5.\n\n"
        + table_to_markdown(ret, "{:.3f}")
        + "\n\n"
        + table_to_markdown(head)
    )


def _quality(c: dict[str, Any]) -> str:
    chk = _df(c["tables"]["checks"])
    chk = chk.rename(columns={chk.columns[0]: "check"}) if "check" not in chk else chk
    show = chk[["check", "precision", "recall", "false_alarm_rate_clean"]].copy()
    show.columns = ["Check", "Precision", "Recall", "Clean photos flagged"]
    show["Clean photos flagged"] = show["Clean photos flagged"].map(_pct)
    cons = c["tables"]["consistency"]
    ctab = pd.DataFrame(cons["test"]).T[["precision", "recall", "f1"]].reset_index()
    ctab.columns = ["Attribute", "Precision", "Recall", "F1"]
    variants = pd.DataFrame(cons["test_variants"]).T[["precision", "recall", "f1"]].reset_index()
    variants.columns = ["Signal", "Precision", "Recall", "F1"]
    variants["Signal"] = variants["Signal"].map(
        {"clf": "classifier alone", "clip": "CLIP alone", "or": "either", "and": "both (used)"}
    )
    b = c["tables"]["batch"]
    return (
        "Image checks: 500 clean test photos plus 8,500 degraded copies we made (blur, dark, washed-out, shrunk, stretched, cluttered, "
        "text). Precision is strict: a flag raised by a different defect type counts as wrong. Overexposure recall is low because white "
        "products on white backgrounds look washed out to a simple measure.\n\n"
        + table_to_markdown(show)
        + "\n\n"
        "Image-text mismatch detection: 300 injected mismatches (100 colour, 100 product type, 100 gender) in the 6,339 test "
        "listings. Precision counts untouched listings as false alarms, and the catalog contains real label errors, so it is a "
        "lower bound.\n\n"
        + table_to_markdown(ctab)
        + "\n\n"
        + table_to_markdown(variants)
        + f"\n\nBatch report over all {b['n_listings']:,} test listings: {b['pct_with_any_issue']:.1f}% carry a flag, "
        f"{b['pct_needs_attention']:.1f}% score below 85, mean score {b['score_mean']:.1f}."
    )


def _system(system: dict[str, Any], disk: dict[str, Any]) -> str:
    rows = []
    for mode, s in system["summary"].items():
        label = {"no_rag": "no-RAG", "rag_with_rewrite": "RAG + guardrail rewrite"}.get(mode, mode)
        rows.append(
            {
                "Pipeline": label,
                "Quality check (s)": f"{s['quality_s']['mean']:.2f}",
                "Attributes (s)": f"{s['classify_s']['mean']:.3f}",
                "Retrieval + LLM + guardrails (s)": f"{s['generate_s']['mean']:.1f}",
                "Total mean (s)": f"{s['total_s']['mean']:.1f}",
                "Total median (s)": f"{s['total_s']['median']:.1f}",
                "Total p95 (s)": f"{s['total_s']['p95']:.1f}",
            }
        )
    ram = system["ram"]
    parts = pd.DataFrame([{"Folder": k, "GB": v} for k, v in disk["parts_gb"].items()])
    big = pd.DataFrame(
        [{"Largest items": k, "GB": v} for k, v in list(disk["large_items_gb"].items())[:6]]
    )
    return (
        f"Photo -> quality check -> attributes -> full listing (title, 5 bullets, description, tags), CPU only, on "
        f"{system['n_photos']} test photos per pipeline with an **uncached** LLM. Models were loaded once "
        f"({system['load_models_s']:.0f} s) and the first LLM call took {system['cold_start_s']:.0f} s (model load).\n\n"
        + table_to_markdown(pd.DataFrame(rows))
        + f"\n\n**Peak RAM:** {ram['combined_mb'] / 1000:.1f} GB in total ({ram['python_mb'] / 1000:.1f} GB the Python process with "
        f"all models loaded + {ram['ollama_mb'] / 1000:.1f} GB the Ollama server holding the language model); the laptop has "
        f"{ram['machine_total_mb'] / 1000:.1f} GB.\n\n"
        f"**Disk footprint:** the project folder is {disk['project_total_gb']:.2f} GB against a budget of {disk['budget_gb']:.0f} GB. "
        f"The language model CatalogIQ uses (Ollama stores it outside the folder) adds {disk.get('ollama_used_gb', 0):.1f} GB, "
        f"so the whole system is {disk.get('total_with_ollama_gb', disk['project_total_gb']):.2f} GB"
        + (
            " (over the budget if the model is counted)"
            if disk.get("total_with_ollama_gb", 0) > disk["budget_gb"]
            else ""
        )
        + (
            "; the Ollama store also holds other models that CatalogIQ does not use ("
            + ", ".join(f"{m['name']} {m['gb']:.1f} GB" for m in disk.get("ollama_other", []))
            + ")"
            if disk.get("ollama_other")
            else ""
        )
        + f"; budget {disk['budget_gb']:.0f} GB.\n\n"
        + table_to_markdown(parts, "{:.2f}")
        + "\n\n"
        + table_to_markdown(big, "{:.2f}")
    )


def _drift(drift: pd.DataFrame, notes: list[dict[str, str]]) -> str:
    counts = drift["status"].value_counts().to_dict()
    show = drift.copy()
    show["explanation"] = [
        next((n["note"] for n in notes if n["match"] in m), "") if s in ("DRIFT", "timing") else ""
        for m, s in zip(show["metric"], show["status"], strict=True)
    ]
    interesting = show[show["status"] != "match"]
    text = (
        f"{len(drift)} numbers recorded by the earlier phases were compared with this re-run: "
        + ", ".join(f"{v} {k}" for k, v in counts.items())
        + ". `match` = within the precision the phase recorded; `timing` = a wall-clock number inside a wide band "
        "(machine state changes timings); `DRIFT` = outside the tolerance.\n\n"
    )
    if len(interesting):
        text += (
            table_to_markdown(
                interesting[["metric", "previous", "now", "diff", "status", "explanation"]],
                "{:.4g}",
            )
            + "\n"
        )
    else:
        text += "Every number matched."
    return (
        text
        + "\n<details><summary>All compared numbers</summary>\n\n"
        + table_to_markdown(
            drift[["metric", "previous", "now", "diff", "status", "source"]], "{:.4g}"
        )
        + "\n\n</details>"
    )


def _impact(impact: dict[str, Any]) -> str:
    inp = pd.DataFrame(impact["inputs"])[["input", "value", "unit", "status", "basis"]]
    rows = pd.DataFrame(impact["rows"])[["quantity", "value", "status", "how"]]
    warn = (
        "**The owner's timings of manual listings have been supplied, so the manual time below is MEASURED.**"
        if impact["timings_supplied"]
        else "**The owner's timings of manual listings have NOT been supplied yet.** Every time and volume below is a placeholder "
        "ASSUMPTION, so every derived business number is ILLUSTRATIVE and must not be quoted as a result. Time 5 manual "
        "listings with `reports/phase7/listing_timing.html`, save the CSV as `reports/phase7/manual_listing_times.csv`, "
        "and run `python -m catalogiq.eval.run_all --render-only`."
    )
    return (
        warn
        + "\n\n**Inputs**\n\n"
        + table_to_markdown(inp, "{:,.1f}")
        + "\n\n**Conversions**\n\n"
        + table_to_markdown(rows)
        + "\n\n**Not estimated (no data to support it):**\n"
        + "\n".join(f"- {x}" for x in impact["not_estimated"])
    )


def _provenance(results: dict[str, Any]) -> str:
    rows = []
    for name, c in results["components"].items():
        p = c["provenance"]
        rows.append(
            {
                "Component": name,
                "Seconds": f"{c['seconds']:.0f}",
                "Recomputed": p.get("recomputed", ""),
                "Cached inputs / replayed": p.get("cached_inputs", p.get("llm", "")),
                "LLM calls that missed the cache": p.get("llm_calls_that_missed_cache", ""),
                "Caveat": p.get("labels", p.get("human_input", "")),
            }
        )
    fixed = get_eval_config()["fixed_sets"]
    sets = "\n".join(f"- **{k}**: {v}" for k, v in fixed.items())
    return "Fixed test sets:\n" + sets + "\n\n" + table_to_markdown(pd.DataFrame(rows))


def _safe(fn: Any) -> str:
    """Section text, or a placeholder when its component has not been run."""
    try:
        return fn()
    except (KeyError, TypeError, ValueError, IndexError):
        return "_Not available: this part was not part of the saved run. Re-run `python -m catalogiq.eval.run_all`._"


def render(results: dict[str, Any], notes: list[dict[str, str]] | None = None) -> str:
    """Build the scorecard markdown from a results dict (see ``run_all``)."""
    g, comp = results["git"], results["components"]
    drift = _df(results["drift"])
    limitations = resolve_path(get_eval_config()["paths"]["limitations"]).read_text(
        encoding="utf-8"
    )
    parts = [
        "# CatalogIQ scorecard",
        f"Generated {results['generated']} from commit `{g['short']}`{' (uncommitted changes present)' if g['dirty'] else ''}. "
        + (
            f"The evaluations ran on commit `{results['run_commit'][:7]}` plus working-tree fixes that were committed afterwards "
            f"(search and the system test were re-run after two measuring bugs were fixed). "
            if results.get("run_commit") and results["run_commit"][:7] != g["short"]
            else ""
        )
        "Produced by `python -m catalogiq.eval.run_all`; every table below was recomputed in that run unless the "
        "Provenance section says otherwise. Experiment `catalogiq-eval` in MLflow, tagged with the commit hash.",
        "## 1. Vision: product attributes from a photo (Phase 2)",
        _safe(lambda: _vision(comp["vision"])),
        "## 2. Visual search and duplicates (Phase 3)",
        _safe(lambda: _search(comp["search"], comp["duplicates"])),
        "## 3. Listing generator: RAG vs no-RAG (Phase 4)",
        _safe(lambda: _listing(comp["listing"])),
        "## 4. Review Q&A (Phase 5)",
        _safe(lambda: _qa(comp["qa"])),
        "## 5. Catalog quality checker (Phase 6)",
        _safe(lambda: _quality(comp["quality"])),
        "## 6. System: end-to-end latency, memory and disk",
        _safe(lambda: _system(results["system"], results["disk"])),
        "## 7. Drift check against the earlier phases",
        _safe(lambda: _drift(drift, notes or [])),
        "## 8. Product impact (assumptions are labelled)",
        _safe(lambda: _impact(results["impact"])),
        "## 9. Limitations and risks",
        limitations,
        "## 10. Provenance",
        _safe(lambda: _provenance(results)),
    ]
    return "\n\n".join(parts) + "\n"
