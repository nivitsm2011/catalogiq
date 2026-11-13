"""Business logic behind the endpoints. Each function takes the loaded ``AppState``."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

from api.state import AppState
from api.validation import ApiError
from catalogiq.config import get_config, get_search_config, resolve_path

MODES = {
    "no_rag": (False, False),
    "rag": (True, False),
    "no_rag_fix": (False, True),
    "rag_fix": (True, True),
}
FILTER_COLUMNS = ("gender", "baseColour", "articleType", "usage", "season", "masterCategory")
CONSISTENCY_ATTRIBUTES = ("baseColour", "articleType", "gender")
MAX_MODEL_SIDE = 1024


def shrink(image: Image.Image, side: int = MAX_MODEL_SIDE) -> Image.Image:
    """Downscale a large photo for the models (they resize to a few hundred pixels anyway)."""
    if max(image.size) <= side:
        return image
    scale = side / max(image.size)
    return image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)


def _clean(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return (
        None
        if value is pd.NA or (not isinstance(value, (list, dict)) and pd.isna(value))
        else value
    )


def item_card(
    row: Any, score: float | None = None, role: str | None = None, reasons: str | None = None
) -> dict[str, Any]:
    """Catalog row -> the card shown in result lists."""
    g = row.get if hasattr(row, "get") else (lambda k, d=None: getattr(row, k, d))
    pid = int(g("id"))
    return {
        "id": pid,
        "title": str(g("productDisplayName") or f"Item {pid}"),
        "articleType": _clean(g("articleType")),
        "baseColour": _clean(g("baseColour")),
        "gender": _clean(g("gender")),
        "usage": _clean(g("usage")),
        "season": _clean(g("season")),
        "masterCategory": _clean(g("masterCategory")),
        "score": _clean(score if score is not None else g("score")),
        "image_url": f"/catalog/image/{pid}",
        "role": _clean(role if role is not None else g("role")),
        "reasons": _clean(reasons if reasons is not None else g("reasons")),
    }


# ---------------------------------------------------------------------------------- attributes
def predict_attributes(state: AppState, image: Image.Image) -> dict[str, Any]:
    """Predict every attribute with calibrated confidence and the runner-up values."""
    from catalogiq.quality.consistency import classifier_probabilities

    start = time.perf_counter()
    probs = classifier_probabilities([np.asarray(shrink(image))], state.predictor)
    out: dict[str, Any] = {}
    for target in state.predictor.targets:
        p = probs[target][0]
        order = np.argsort(-p)
        classes = state.labels[target]
        threshold = float(state.predictor.threshold[target])
        out[target] = {
            "label": classes[int(order[0])],
            "confidence": float(p[order[0]]),
            "needs_review": bool(p[order[0]] < threshold),
            "threshold": threshold,
            "alternatives": [
                {"label": classes[int(i)], "probability": float(p[i])} for i in order[1:5]
            ],
        }
    return {
        "attributes": out,
        "image": {"width": image.width, "height": image.height},
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


# --------------------------------------------------------------------------------------- listing
NICE_ATTR = {
    "gender": "gender",
    "articleType": "product type",
    "baseColour": "colour",
    "usage": "occasion",
}


def friendly_style(code: str) -> str:
    """Plain-English version of a style-guard code such as ``tag_count:3``."""
    from catalogiq.config import get_rag_config

    st = get_rag_config()["style"]
    name, _, val = code.partition(":")
    if name == "title_length":
        return f"The title has {val} characters; aim for {st['title_min_chars']} to {st['title_max_chars']}."  # noqa: E501
    if name == "description_words":
        return f"The description has {val} words; aim for {st['description_min_words']} to {st['description_max_words']}."  # noqa: E501
    if name == "tag_count":
        return f"There are {val} search tags; aim for {st['tags_min']} to {st['tags_max']}."
    if name == "bullet_too_long":
        return f"A bullet point is longer than {st['bullet_max_chars']} characters."
    if name == "brand_not_first":
        return "Start the title with the brand."
    if name == "title_order":
        return "Put the gender before the colour in the title."
    if name.startswith("title_missing_"):
        return f"The title should mention the {NICE_ATTR.get(name[14:], name[14:])}."
    if name.startswith("uncertain_") and name.endswith("_in_title"):
        attr = name[10:-9]
        return f"The title states the {NICE_ATTR.get(attr, attr)}, but the photo reading is uncertain: please confirm it."  # noqa: E501
    return code


def generate_listing(
    state: AppState, attributes: dict[str, dict[str, Any]], notes: dict[str, str], mode: str
) -> dict[str, Any]:
    """Generate a listing from (possibly edited) attributes; one LLM call at a time."""
    if "articleType" not in attributes:
        raise ApiError(
            422,
            "missing_attribute",
            "The attributes must include 'articleType' (the product type).",
        )
    unknown = [k for k in attributes if k not in state.labels]
    if unknown:
        raise ApiError(
            422,
            "unknown_attribute",
            f"Unknown attribute(s) {unknown}; known: {sorted(state.labels)}.",
        )
    attrs = {
        k: (v["label"], float(v.get("confidence", 1.0)), bool(v.get("needs_review", False)))
        for k, v in attributes.items()
    }
    use_rag, fix = MODES[mode]
    start = time.perf_counter()
    with state.llm_lock:
        result = state.generator.generate(attrs, notes, use_rag, fix)
    elapsed = time.perf_counter() - start
    rep = result.report
    return {
        "listing": result.listing.model_dump() if result.listing else None,
        "guardrails": (
            None
            if rep is None
            else {
                "hallucinated": rep.hallucinated,
                "unsupported": [{"kind": k, "term": t} for k, t in rep.unsupported],
                "banned": list(rep.banned),
                "style_violations": list(rep.style_violations),
                "style_messages": [friendly_style(v) for v in rep.style_violations],
                "style_compliant": rep.style_compliant,
                "coverage": float(rep.coverage),
                "missing_attributes": list(rep.missing_attributes),
            }
        ),
        "group": result.group,
        "mode": mode,
        "attempts": result.attempts,
        "fixed": result.fixed,
        "retrieved": list(result.retrieved),
        "error": result.error,
        "elapsed_s": elapsed,
        "generation_s": float(result.latency_s),
    }


# ----------------------------------------------------------------------------------------- search
def check_filters(state: AppState, filters: dict[str, str | None]) -> dict[str, str]:
    """Drop empty filters and reject values that do not exist in the catalog."""
    clean = {k: v for k, v in filters.items() if v}
    for key, value in clean.items():
        if key not in FILTER_COLUMNS:
            raise ApiError(
                422, "unknown_filter", f"Unknown filter '{key}'; use one of {list(FILTER_COLUMNS)}."
            )
        known = set(state.catalog[key].dropna().astype(str))
        if value not in known:
            raise ApiError(
                422,
                "unknown_filter_value",
                f"No item has {key} = '{value}'. Examples: {sorted(known)[:12]}.",
            )
    return clean


def _cards(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [item_card(r) for r in frame.to_dict("records")]


def search_image(
    state: AppState, image: Image.Image, k: int, filters: dict[str, str], exclude_id: int | None
) -> dict[str, Any]:
    """Visually similar catalog items for a photo."""
    start = time.perf_counter()
    extra = 1 if exclude_id is not None else 0
    res = state.engine.search_by_image(shrink(image), k + extra, filters or None)
    if exclude_id is not None:
        res = res[res["id"] != exclude_id]
    res = res.head(k)
    return {
        "query": "photo",
        "k": k,
        "filters": filters,
        "results": _cards(res),
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


def search_text(state: AppState, q: str, k: int, filters: dict[str, str]) -> dict[str, Any]:
    """Catalog items matching a text query (CLIP text -> image)."""
    start = time.perf_counter()
    res = state.engine.search_by_text(q, k, filters or None)
    return {
        "query": q,
        "k": k,
        "filters": filters,
        "results": _cards(res),
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


def shop_the_look(
    state: AppState, item_id: int | None, image: Image.Image | None
) -> dict[str, Any]:
    """Complementary items for a catalog item or a photo."""
    start = time.perf_counter()
    if (item_id is None) == (image is None):
        raise ApiError(
            400, "ambiguous_input", "Send exactly one of 'item_id' or a photo in 'file'."
        )
    if item_id is not None and int(item_id) not in state.engine.id_to_pos:
        raise ApiError(404, "unknown_item", f"No catalog item with id {item_id}.")
    look = state.looks.recommend(int(item_id) if item_id is not None else shrink(image))  # type: ignore[arg-type]
    items = look.items
    note = None
    if items.empty:
        note = "This product type is not part of any outfit, so there is nothing to pair it with."
    query = {k: _clean(v) for k, v in look.query.items()}
    return {
        "query": query,
        "items": [] if items.empty else _cards(items),
        "note": note,
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


# --------------------------------------------------------------------------------------- quality
def quality_check(
    state: AppState, image: Image.Image, stated: dict[str, str], title: str | None
) -> dict[str, Any]:
    """Photo checks, photo-vs-listing consistency, completeness and the 0-100 score."""
    from catalogiq.quality.clip_checks import text_margin
    from catalogiq.quality.completeness import completeness
    from catalogiq.quality.consistency import (
        classifier_probabilities,
        clip_probabilities,
        flag_mismatch,
        load_consistency_thresholds,
        mismatch_signals,
    )
    from catalogiq.quality.image_checks import check_image, load_thresholds
    from catalogiq.quality.score import listing_score

    start = time.perf_counter()
    arr = np.asarray(image)
    small = np.asarray(shrink(image))
    emb = state.encoder.encode_image(small)
    margin = float(text_margin(emb, state.text_prompts)[0])
    check = check_image(arr, load_thresholds(), text_margin=margin)
    cth = load_consistency_thresholds()
    probs = classifier_probabilities([small], state.predictor)
    mismatches = []
    for a in CONSISTENCY_ATTRIBUTES:
        value = stated.get(a)
        if not value or value not in state.labels[a]:
            continue
        idx = np.array([state.labels[a].index(value)])
        sig = mismatch_signals(idx, probs[a], clip_probabilities(emb[None], state.value_emb[a]))
        if flag_mismatch(sig, cth["min_conf"], cth["margin"], cth["mode"])[0]:
            use_clf = sig["clf_alt_conf"].iat[0] >= cth["min_conf"]
            seen = state.labels[a][
                int(sig["clf_top"].iat[0] if use_clf else sig["clip_best_alt"].iat[0])
            ]
            mismatches.append({"attribute": a, "stated": value, "seen": seen})
    listing = {**stated, "productDisplayName": title or stated.get("productDisplayName")}
    comp = completeness(listing)
    result = listing_score(check["issues"], mismatches, comp["pct"], comp["missing"])
    return {
        "score": result["score"],
        "components": result["components"],
        "needs_attention": result["needs_attention"],
        "flags": result["flags"],
        "image_issues": check["issues"],
        "image_metrics": {k: float(v) for k, v in check["metrics"].items()},
        "mismatches": mismatches,
        "completeness": {k: comp[k] for k in ("group", "required", "missing", "pct")},
        "latency_ms": 1000 * (time.perf_counter() - start),
    }


# ----------------------------------------------------------------------------------------- reviews
def review_products(state: AppState) -> list[dict[str, Any]]:
    """Reviewed products: the demo ones first (instant answers), then the biggest."""
    by_item: dict[str, list[str]] = {}
    for q in state.questions:
        by_item.setdefault(q["item"], []).append(q["q"])
    counts = state.chunks.drop_duplicates("review_id").groupby("item_id").size()
    items = state.review_items.set_index("parent_asin")
    rows = []
    for asin, n in counts.items():
        if asin not in items.index:
            continue
        it = items.loc[asin]
        rows.append(
            {
                "item_id": asin,
                "title": str(it["title"]),
                "store": _clean(it.get("store")),
                "average_rating": _clean(it.get("average_rating")),
                "n_reviews": int(n),
                "demo": asin in by_item,
                "suggested_questions": by_item.get(asin, [])[:4],
            }
        )
    rows.sort(key=lambda r: (not r["demo"], -r["n_reviews"]))
    return rows[: 20 + 40]


def _require_review_item(state: AppState, item_id: str) -> pd.Series:
    items = state.review_items.set_index("parent_asin")
    if item_id not in items.index:
        raise ApiError(404, "unknown_product", f"No reviewed product with id '{item_id}'.")
    return items.loc[item_id]


def ask_reviews(state: AppState, item_id: str, question: str) -> dict[str, Any]:
    """Answer a question from one product's reviews, with citations, or refuse."""
    _require_review_item(state, item_id)
    start = time.perf_counter()
    with state.llm_lock:
        res = state.qa.answer(item_id, question)
    out = res.output
    texts = state.review_text
    citations = []
    for rid in out.citations:
        meta = res.evidence[res.evidence["review_id"] == rid].head(1)
        row = meta.iloc[0] if len(meta) else None
        full = (
            str(texts.at[rid, "full"])
            if rid in texts.index
            else (str(row["text"]) if row is not None else "")
        )
        citations.append(
            {
                "review_id": rid,
                "rating": _clean(row["rating"]) if row is not None else None,
                "verified_purchase": bool(row["verified_purchase"]) if row is not None else None,
                "helpful_vote": int(row["helpful_vote"]) if row is not None else None,
                "date": str(row["date"]) if row is not None and "date" in row else None,
                "text": full,
            }
        )
    return {
        "answer": out.answer,
        "refused": out.refused,
        "confidence": out.confidence,
        "aspect": out.aspect,
        "evidence_count": out.evidence_count,
        "citations": citations,
        "reviewers_agree": out.reviewers_agree,
        "reviewers_disagree": out.reviewers_disagree,
        "top_similarity": float(res.top_similarity),
        "elapsed_s": time.perf_counter() - start,
        "invalid_citations": list(res.invalid_citations),
        "error": res.error,
    }


def review_summary(state: AppState, item_id: str, phrase: bool) -> dict[str, Any]:
    """Aspect summary: numbers from code; the sentence is a template or, on request, LLM-phrased."""
    from catalogiq.rag.summary import (
        LABELS,
        phrase_summary,
        product_facts,
        tag_review_aspects,
        template_summary,
    )

    it = _require_review_item(state, item_id)
    sub = state.chunks[state.chunks["item_id"] == item_id].reset_index(drop=True)
    review_aspects = tag_review_aspects(sub)
    texts = state.review_text
    ids = review_aspects["review_id"].tolist()
    text_of = dict(zip(ids, texts.loc[ids, "full"].tolist(), strict=True))
    facts = product_facts(item_id, review_aspects, text_of)
    if phrase:
        with state.llm_lock:
            phrased = phrase_summary(facts, str(it["title"]), state.llm)
        summary, source = phrased["summary"], phrased["source"]
    else:
        summary, source = template_summary(facts), "template"
    return {
        "item_id": item_id,
        "title": str(it["title"]),
        "n_reviews": int(facts["n_reviews"]),
        "aspects": facts["aspects"],
        "labels": LABELS,
        "summary": summary,
        "source": source,
    }


# -------------------------------------------------------------------------------------- ops + demo
def _quality_report(state: AppState) -> pd.DataFrame:
    def build() -> pd.DataFrame:
        path = resolve_path(state.quality_paths()["batch_csv"])
        if not path.exists():
            raise ApiError(
                404,
                "no_quality_report",
                "Run scripts/run_quality_report.py to create the batch report.",
            )
        df = pd.read_csv(path).fillna("")
        names = state.catalog.set_index("id")["productDisplayName"]
        df["title"] = df["id"].map(names).fillna("")
        return df

    return state.lazy("quality_report", build)


def ops_summary(state: AppState, category: str | None, limit: int) -> dict[str, Any]:
    """KPIs, issue counts, by-category table and the worst listings of the batch quality report."""
    df = _quality_report(state)
    has_issue = (df["image_issues"] != "") | (df["mismatches"] != "") | (df["missing"] != "")
    types = pd.concat(
        [
            df["image_issues"].str.split(";").explode(),
            df["mismatches"].str.split(";").explode().map(lambda a: f"mismatch_{a}" if a else ""),
            df["missing"].str.split(";").explode().map(lambda a: f"missing_{a}" if a else ""),
        ]
    )
    types = types[types != ""].value_counts()
    by_cat = (
        df.assign(issue=has_issue)
        .groupby("masterCategory")
        .agg(
            n=("id", "size"),
            pct_with_issue=("issue", lambda s: 100 * s.mean()),
            mean_score=("score", "mean"),
        )
        .round(2)
        .reset_index()
        .to_dict("records")
    )
    by_group = (
        df.assign(issue=has_issue)
        .groupby("group")
        .agg(
            n=("id", "size"),
            pct_with_issue=("issue", lambda s: 100 * s.mean()),
            mean_score=("score", "mean"),
        )
        .round(2)
        .reset_index()
        .sort_values("pct_with_issue", ascending=False)
        .to_dict("records")
    )
    pool = (
        df if not category else df[(df["masterCategory"] == category) | (df["group"] == category)]
    )
    worst = pool.sort_values(["score", "id"]).head(limit)
    return {
        "n_listings": int(len(df)),
        "avg_score": float(df["score"].mean()),
        "pct_with_issue": float(100 * has_issue.mean()),
        "pct_needs_attention": float(100 * df["needs_attention"].astype(str).eq("True").mean()),
        "issue_counts": {k: int(v) for k, v in types.items()},
        "by_master_category": by_cat,
        "by_group": by_group,
        "score_hist": np.histogram(df["score"], bins=[0, 50, 60, 70, 80, 85, 90, 95, 100.01])[
            0
        ].tolist(),
        "worst": [
            {
                **item_card(
                    {
                        "id": r["id"],
                        "productDisplayName": r["title"],
                        "articleType": r["articleType"],
                        "masterCategory": r["masterCategory"],
                    }
                ),
                "quality_score": float(r["score"]),
                "issues": [x for x in str(r["image_issues"]).split(";") if x],
                "mismatches": [x for x in str(r["mismatches"]).split(";") if x],
                "missing": [x for x in str(r["missing"]).split(";") if x],
                "fixes": [_friendly_fix(x) for x in str(r["fixes"]).split(" | ") if x],
            }
            for r in worst.to_dict("records")
        ],
    }


def _friendly_fix(text: str) -> str:
    """The batch report was written with field names; show them as words."""
    for raw, nice in (("baseColour", "colour"), ("articleType", "product type")):
        text = text.replace(raw, nice)
    return text


def _decisions_path() -> Path:
    return resolve_path("data/interim/duplicate_decisions.json")


def _load_decisions() -> dict[str, str]:
    path = _decisions_path()
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def duplicate_queue(state: AppState, limit: int) -> dict[str, Any]:
    """Flagged duplicate pairs (CLIP >= tuned threshold and pHash <= 8), clusters and decisions."""
    from catalogiq.search.duplicates import DuplicateFinder, duplicate_groups, load_phash

    def build() -> pd.DataFrame:
        finder = DuplicateFinder(
            state.catalog, state.image_emb, load_phash(state.catalog["id"]), min_similarity=0.9
        )
        thr_path = resolve_path("reports/duplicate_threshold.json")
        threshold = (
            json.loads(thr_path.read_text(encoding="utf-8"))["recommended_threshold"]
            if thr_path.exists()
            else 0.975
        )
        return finder.find_duplicates(
            threshold, get_search_config()["duplicates"]["phash_max_distance"]
        )

    pairs = state.lazy("duplicate_pairs", build)
    groups = duplicate_groups(pairs) if len(pairs) else {}
    decisions = _load_decisions()
    sizes = pd.Series(list(groups.values())).value_counts() if groups else pd.Series(dtype=int)
    names = state.catalog.set_index("id")
    rows = []
    for r in pairs.head(limit * 3).to_dict("records"):
        key = f"{int(r['id_a'])}-{int(r['id_b'])}"
        a, b = names.loc[int(r["id_a"])], names.loc[int(r["id_b"])]
        rows.append(
            {
                "id_a": int(r["id_a"]),
                "id_b": int(r["id_b"]),
                "title_a": str(a["productDisplayName"]),
                "title_b": str(b["productDisplayName"]),
                "similarity": float(r["similarity"]),
                "phash_distance": float(r["phash_distance"]),
                "image_a": f"/catalog/image/{int(r['id_a'])}",
                "image_b": f"/catalog/image/{int(r['id_b'])}",
                "decision": decisions.get(key),
            }
        )
    open_rows = [r for r in rows if r["decision"] is None][:limit]
    return {
        "n_pairs": int(len(pairs)),
        "n_products": int(len(groups)),
        "n_clusters": int(len(set(groups.values()))) if groups else 0,
        "largest_cluster": int(sizes.max()) if len(sizes) else 0,
        "n_decided": len(decisions),
        "pairs": open_rows,
    }


def save_duplicate_decision(id_a: int, id_b: int, decision: str) -> dict[str, Any]:
    """Persist the catalog team's decision for a flagged pair."""
    decisions = _load_decisions()
    decisions[f"{id_a}-{id_b}"] = decision
    path = _decisions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=1), encoding="utf-8")
    return {"saved": True, "n_decided": len(decisions)}


def catalog_image_path(item_id: int) -> Path:
    """Path of a catalog photo; raises 404 when it does not exist."""
    from catalogiq.data.fashion import image_path

    path = image_path(resolve_path(get_config()["data"]["fashion"]["images_dir"]), int(item_id))
    if not path.exists():
        raise ApiError(404, "unknown_item", f"No photo for catalog item {item_id}.")
    return path


def warm_up(state: AppState) -> None:
    """Touch the lazily built parts (retriever, review texts, duplicate pairs, quality report).

    Runs in a background thread after startup so the first request of each page is quick.
    """
    from catalogiq.logging_utils import get_logger

    log = get_logger("api.warmup")
    steps = (
        ("retriever", lambda: state.generator.retriever.search("shirt", k=1, group=None)),
        ("review texts", lambda: state.review_text),
        ("quality report", lambda: _quality_report(state)),
        ("duplicate pairs", lambda: duplicate_queue(state, 1)),
    )
    for name, step in steps:
        try:
            step()
            log.info("warmed %s", name)
        except Exception as exc:  # noqa: BLE001 - warming is best effort
            log.warning("could not warm %s: %s", name, exc)
