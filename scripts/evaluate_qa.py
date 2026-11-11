"""Phase 5 evaluation: retrieval variants, evidence-gate tuning, answers, faithfulness, summaries.

Resumable stages writing to reports/phase5/. Needs the review index (scripts/build_review_index.py)
and, for generation/judging/summaries, a running Ollama.
Usage: python scripts/evaluate_qa.py retrieval | generate | judge | summaries | all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_qa_config, resolve_path
from catalogiq.llm.client import get_llm_client
from catalogiq.logging_utils import get_logger
from catalogiq.rag.aspects import tag_with
from catalogiq.rag.eval_qa import (
    best_gate_threshold,
    gate_rates,
    ground_truth,
    grouped_cv_gate,
    judge_answer,
    load_questions,
    recall_at_k,
    review_texts,
    validate_questions,
)
from catalogiq.rag.knowledge_base import get_client
from catalogiq.rag.qa import ReviewQA
from catalogiq.rag.reviews_index import load_chunks
from catalogiq.rag.summary import (
    phrase_summary,
    product_facts,
    tag_review_aspects,
    template_summary,
)
from catalogiq.utils import set_seed

logger = get_logger("evaluate_qa")
VARIANTS = {
    "bm25": ("bm25", False),
    "vector": ("vector", False),
    "hybrid": ("hybrid", False),
    "hybrid_rerank": ("hybrid", True),
}


def tune_gate(
    rows: pd.DataFrame, grid: np.ndarray, min_supporting: int
) -> tuple[float, pd.DataFrame]:
    """Similarity threshold maximising balanced accuracy of the gate on the DEV questions."""

    def passes(pairs: list[tuple[str, float]], t: float) -> bool:
        return len({r for r, s in pairs if s >= t}) >= min_supporting

    res = []
    for t in grid:
        for part, sub in (("dev", rows[rows["dev"]]), ("test", rows[~rows["dev"]])):
            ans, una = sub[sub["kind"] == "answerable"], sub[sub["kind"] == "unanswerable"]
            tpr = float(np.mean([passes(p, t) for p in ans["pairs"]]))
            tnr = float(np.mean([not passes(p, t) for p in una["pairs"]]))
            res.append(
                {
                    "threshold": round(float(t), 3),
                    "part": part,
                    "answer_rate": tpr,
                    "refuse_rate": tnr,
                    "balanced_acc": (tpr + tnr) / 2,
                }
            )
    table = pd.DataFrame(res)
    dev = table[table["part"] == "dev"].sort_values(
        ["balanced_acc", "threshold"], ascending=[False, True]
    )
    return float(dev.iloc[0]["threshold"]), table


def stage_retrieval(qa: ReviewQA, gt: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Recall@5 of every retrieval variant, then tune the evidence gate on the dev questions."""
    rows = []
    for q in gt.itertuples():
        pattern = getattr(q, "evidence", None)
        for name, (mode, rerank) in VARIANTS.items():
            t0 = time.perf_counter()
            ev = qa.retrieve(q.item, q.q, mode=mode, rerank=rerank)
            ms = 1000 * (time.perf_counter() - t0)
            texts = ev["text"].tolist()
            answerable = q.kind == "answerable"
            first = next(
                (i + 1 for i, t in enumerate(texts) if answerable and recall_at_k([t], pattern, 1)),
                0,
            )
            rows.append(
                {
                    "id": q.id,
                    "kind": q.kind,
                    "dev": q.dev,
                    "variant": name,
                    "item": q.item,
                    "recall5": recall_at_k(texts, pattern, 5) if answerable else np.nan,
                    "mrr": (1 / first if first else 0.0) if answerable else np.nan,
                    "top_sim": float(ev["similarity"].max()) if len(ev) else 0.0,
                    "pairs": list(zip(ev["review_id"], ev["similarity"], strict=True)),
                    "latency_ms": ms,
                }
            )
        logger.info("retrieval %s done", q.id)
    df = pd.DataFrame(rows)
    df.to_pickle(out / "retrieval.pkl")
    final = df[df["variant"] == "hybrid_rerank"]
    grid = np.round(np.arange(0.10, 0.71, 0.02), 3)
    qcfg = get_qa_config()
    ms = qcfg["retrieval"]["min_supporting"]
    # (1) the pre-specified plan: tune on the dev questions, look at the test questions
    t_dev, table = tune_gate(final, grid, ms)
    table.to_csv(out / "gate_tuning.csv", index=False)
    # (2) what we do instead: product-grouped cross-validation, final threshold fit on all 60
    cv = grouped_cv_gate(final, grid, ms, qcfg["evaluation"]["cv_folds"], qcfg["seed"])
    t_all, bal_all = best_gate_threshold(final, grid, ms)
    a_all, u_all = gate_rates(final, t_all, ms)
    gate = {
        "min_similarity": t_all,
        "fit_on_all_60": {"answer_rate": a_all, "refuse_rate": u_all, "balanced_acc": bal_all},
        "cross_validated": cv,
        "dev_tuned_threshold": t_dev,
        "dev_tuned_dev_to_test": {
            part: gate_rates(final[final["dev"] == (part == "dev")], t_dev, ms)
            for part in ("dev", "test")
        },
    }
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    logger.info("gate: %s", gate)
    return df


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()}


def stage_generate(qa: ReviewQA, gt: pd.DataFrame, out: Path, reviews: pd.DataFrame) -> None:
    """Answer all 60 questions (resumable)."""
    path = out / "answers.jsonl"
    done = _done_ids(path)
    text_of = dict(zip(reviews["review_id"], reviews["full"], strict=True))
    with path.open("a", encoding="utf-8") as fh:
        for q in gt.itertuples():
            if q.id in done:
                continue
            res = qa.answer(q.item, q.q)
            rec = {
                "id": q.id,
                "kind": q.kind,
                "dev": q.dev,
                "item": q.item,
                "q": q.q,
                "output": res.output.model_dump(),
                "latency_s": res.latency_s,
                "top_similarity": res.top_similarity,
                "invalid_citations": res.invalid_citations,
                "shown": {r: text_of[r][:800] for r in res.evidence["review_id"]},
                "error": res.error,
            }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            logger.info("%s refused=%s %.1fs", q.id, res.output.refused, res.latency_s)


def stage_judge(llm: Any, out: Path, reviews: pd.DataFrame) -> None:
    """Claim-level faithfulness judge on every answer that was not a refusal (resumable)."""
    ans = [json.loads(x) for x in (out / "answers.jsonl").read_text(encoding="utf-8").splitlines()]
    path = out / "faithfulness.jsonl"
    done = _done_ids(path)
    text_of = dict(zip(reviews["review_id"], reviews["full"], strict=True))
    with path.open("a", encoding="utf-8") as fh:
        for a in ans:
            if a["output"]["refused"] or a["id"] in done:
                continue
            j = judge_answer(llm, a["output"], text_of)
            fh.write(json.dumps({"id": a["id"], "judge": j}) + chr(10))
            fh.flush()
            logger.info(
                "judged %s: %d/%d claims supported", a["id"], j["n_supported"], j["n_claims"]
            )


def stage_summaries(
    llm: Any, chunks: pd.DataFrame, reviews: pd.DataFrame, out: Path, titles: dict[str, str]
) -> None:
    """Aspect summaries for the 20 test products with the selected aspect tagger."""
    comp = resolve_path(get_qa_config()["paths"]["aspect_results"])
    method, thr = "rules", None
    if comp.exists():
        data = json.loads(comp.read_text(encoding="utf-8"))
        method, thr = data["chosen"], data.get("thresholds", {}).get(data["chosen"])
    logger.info("aspect method for summaries: %s", method)
    asins = sorted({q["item"] for q in load_questions()})
    sub = chunks[chunks["item_id"].isin(asins)].reset_index(drop=True)
    tags = tag_with(method, sub["text"].tolist(), thr, llm)
    ra = tag_review_aspects(sub, tags)
    text_of = dict(zip(reviews["review_id"], reviews["full"], strict=True))
    res = {}
    for a in asins:
        facts = product_facts(a, ra, text_of)
        phrased = phrase_summary(facts, titles.get(a, a), llm)
        res[a] = {
            "title": titles.get(a, a),
            "facts": facts,
            "template": template_summary(facts),
            **phrased,
        }
        logger.info("summary %s source=%s", a, phrased["source"])
    (out / "summaries.json").write_text(json.dumps({"method": method, "products": res}, indent=1))


def main() -> None:
    """Run one stage (or all) of the Phase 5 evaluation."""
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    cfg = get_qa_config()
    set_seed(cfg["seed"])
    out = resolve_path(cfg["paths"]["eval_dir"])
    out.mkdir(parents=True, exist_ok=True)
    reviews = review_texts()
    gt = ground_truth(load_questions(), reviews)
    problems = validate_questions(gt)
    if problems:
        raise SystemExit("invalid test set: " + "; ".join(problems))
    chunks = load_chunks()
    col = get_client().get_collection(cfg["paths"]["collection"])
    meta = pd.read_parquet(resolve_path(get_config()["data"]["reviews"]["meta_parquet"]))
    titles = dict(zip(meta["parent_asin"], meta["title"], strict=True))
    if stage in ("retrieval", "all"):
        stage_retrieval(ReviewQA(None, chunks, col, titles), gt, out)  # type: ignore[arg-type]
    gate_path = out / "gate.json"
    gate = json.loads(gate_path.read_text())["min_similarity"] if gate_path.exists() else None
    if stage in ("generate", "all"):
        stage_generate(ReviewQA(get_llm_client(), chunks, col, titles, gate), gt, out, reviews)
    if stage in ("judge", "all"):
        stage_judge(get_llm_client(), out, reviews)
    if stage in ("summaries", "all"):
        stage_summaries(get_llm_client(), chunks, reviews, out, titles)


if __name__ == "__main__":
    main()
