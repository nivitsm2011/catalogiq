"""Phase 5 evaluation helpers: test questions, ground truth from regexes, metrics."""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from catalogiq.config import get_config, get_qa_config, resolve_path


def load_questions() -> list[dict[str, Any]]:
    """The 60 test questions from ``configs/qa_test_questions.yaml``."""
    path = resolve_path(get_qa_config()["paths"]["questions"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


def review_texts() -> pd.DataFrame:
    """Reviews with their ``review_id`` (row number) and title + text joined."""
    rv = pd.read_parquet(resolve_path(get_config()["data"]["reviews"]["reviews_parquet"]))
    rv["review_id"] = [f"R{i}" for i in range(len(rv))]
    rv["full"] = rv["title"].fillna("") + ". " + rv["text"].fillna("")
    return rv


def ground_truth(questions: list[dict[str, Any]], reviews: pd.DataFrame) -> pd.DataFrame:
    """Per question: reviews carrying the answer (``evidence``) or the topic (``absent``).

    ``n_matching`` is the number of matching reviews of the product: answerable questions need at
    least 3, unanswerable ones exactly 0 (checked by ``validate_questions``).
    """
    rows = []
    for q in questions:
        rev = reviews[reviews["parent_asin"] == q["item"]]
        pattern = q.get("evidence") or q["absent"]
        hits = rev[rev["full"].str.contains(pattern, case=False, regex=True)]
        rows.append({**q, "n_matching": len(hits), "gold_reviews": list(hits["review_id"])})
    return pd.DataFrame(rows)


def validate_questions(gt: pd.DataFrame, min_answerable: int = 3) -> list[str]:
    """Problems with the test set (empty list = valid)."""
    problems = []
    for r in gt.itertuples():
        if r.kind == "answerable" and r.n_matching < min_answerable:
            problems.append(f"{r.id}: only {r.n_matching} reviews match the evidence regex")
        if r.kind == "unanswerable" and r.n_matching != 0:
            problems.append(f"{r.id}: {r.n_matching} reviews mention the 'unanswerable' topic")
    counts = gt["kind"].value_counts().to_dict()
    if counts != {"answerable": 45, "unanswerable": 15}:
        problems.append(f"expected 45 answerable + 15 unanswerable, got {counts}")
    if gt["item"].nunique() != 20 or len(gt) != 60:
        problems.append("expected 60 questions over 20 products")
    return problems


def chunk_has_answer(chunk_text: str, pattern: str) -> bool:
    """Does a retrieved chunk contain answer-bearing text?"""
    return bool(re.search(pattern, chunk_text, flags=re.IGNORECASE))


def recall_at_k(retrieved_texts: list[str], pattern: str, k: int = 5) -> float:
    """1.0 if any of the first ``k`` retrieved chunks matches the evidence regex, else 0.0."""
    return float(any(chunk_has_answer(t, pattern) for t in retrieved_texts[:k]))


def mean_ci(values: list[float] | np.ndarray, z: float = 1.96) -> tuple[float, float, float]:
    """Mean and a normal-approximation 95% interval (a rough guide for small samples)."""
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    half = z * float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, max(0.0, m - half), min(1.0, m + half)


SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_claims(output: dict[str, Any]) -> list[dict[str, Any]]:
    """Claims of an answer: sentences of the answer, agree and disagree fields.

    Each claim keeps the review ids cited inside it (falling back to all cited reviews of the
    answer). Fragments of fewer than four words ("None mentioned") are not claims.
    """
    from catalogiq.rag.qa import extract_ids

    claims: list[dict[str, Any]] = []
    for field in ("answer", "reviewers_agree", "reviewers_disagree"):
        for sent in SENTENCE_END.split(output.get(field, "") or ""):
            text = re.sub(r"\s*\[[^\[\]]*\]", "", sent).strip()
            if len(text.split()) < 4 or text.lower().startswith("none"):
                continue
            ids = [i for i in extract_ids(sent) if i in output["citations"]]
            claims.append({"field": field, "text": text, "cited": ids or list(output["citations"])})
    return claims


def judge_claim(llm: Any, claim: str, cited_texts: list[str]) -> bool | None:
    """LLM judge for ONE claim: is it supported by the cited review texts? ``None`` if it fails."""

    path = resolve_path(get_qa_config()["paths"]["prompts_faith"])
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    user = p["user"].format(
        claim=claim, cited=chr(10).join(f"- {t}" for t in cited_texts) or "(no cited reviews)"
    )
    schema = {
        "type": "object",
        "properties": {"supported": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["supported"],
    }
    messages = [{"role": "system", "content": p["system"]}, {"role": "user", "content": user}]
    temp = get_qa_config()["evaluation"]["judge_temperature"]
    for _ in range(2):
        try:
            return bool(
                json.loads(llm.chat(messages, schema=schema, temperature=temp).text)["supported"]
            )
        except Exception:  # noqa: BLE001 - retry once, then give up
            continue
    return None


def judge_answer(llm: Any, output: dict[str, Any], review_text: dict[str, str]) -> dict[str, Any]:
    """Claim-level faithfulness: every claim must be supported by the reviews it cites."""
    results = []
    for c in split_claims(output):
        texts = [review_text[i][:800] for i in c["cited"] if i in review_text]
        results.append({**c, "supported": judge_claim(llm, c["text"], texts)})
    judged = [r for r in results if r["supported"] is not None]
    return {
        "claims": results,
        "n_claims": len(judged),
        "n_supported": sum(bool(r["supported"]) for r in judged),
        "faithful": bool(judged) and all(r["supported"] for r in judged),
    }


def answer_metrics(
    answers: pd.DataFrame, faith: pd.DataFrame, gt: pd.DataFrame, review_text: dict[str, str]
) -> dict[str, Any]:
    """End-to-end metrics from the stored answers.

    Args:
        answers: One row per question (kind, output dict, invalid_citations, latency_s).
        faith: Judge results, one row per answered question (``id`` and ``judge`` dict).
        gt: Ground truth indexed by question id (needs ``evidence`` for answerable questions).
        review_text: ``{review_id: text}``.
    """
    a = answers.copy()
    a["refused"] = a["output"].map(lambda o: o["refused"])
    una, ans = a[a["kind"] == "unanswerable"], a[a["kind"] == "answerable"]
    answered = ans[~ans["refused"]]
    kept, relevant = 0, 0
    for r in answered.itertuples():
        pattern = gt.loc[r.id, "evidence"]
        for c in r.output["citations"]:
            kept += 1
            relevant += chunk_has_answer(review_text[c], pattern)
    invalid = int(a["invalid_citations"].map(len).sum())
    total_cited = int(a["output"].map(lambda o: len(o["citations"])).sum()) + invalid
    j = faith[faith["judge"].notna()]
    claims = sum(x["n_claims"] for x in j["judge"])
    return {
        "n_answerable": len(ans),
        "n_unanswerable": len(una),
        "correct_refusal_rate": float(una["refused"].mean()),
        "false_refusal_rate": float(ans["refused"].mean()),
        "answered_answerable": len(answered),
        "unanswerable_answered": int((~una["refused"]).sum()),
        "citations_kept": kept,
        "invalid_citation_ids": invalid,
        "citation_validity": 1 - invalid / max(total_cited, 1),
        "citation_relevance": relevant / max(kept, 1),
        "judged_answers": len(j),
        "faithful_answers": float(np.mean([x["faithful"] for x in j["judge"]])) if len(j) else 0.0,
        "claim_support_rate": sum(x["n_supported"] for x in j["judge"]) / max(claims, 1),
        "latency_mean_s": float(a["latency_s"].mean()),
        "latency_answered_mean_s": float(a[~a["refused"]]["latency_s"].mean()),
        "latency_refused_mean_s": float(a[a["refused"]]["latency_s"].mean()),
    }


def gate_passes(pairs: list[tuple[str, float]], threshold: float, min_supporting: int) -> bool:
    """Evidence gate on (review_id, similarity) pairs: enough DISTINCT reviews above the bar."""
    return len({r for r, sim in pairs if sim >= threshold}) >= min_supporting


def gate_rates(rows: pd.DataFrame, threshold: float, min_supporting: int) -> tuple[float, float]:
    """(answer rate on answerable questions, refusal rate on unanswerable questions)."""
    ans = rows[rows["kind"] == "answerable"]
    una = rows[rows["kind"] == "unanswerable"]
    a = float(np.mean([gate_passes(p, threshold, min_supporting) for p in ans["pairs"]]))
    u = float(np.mean([not gate_passes(p, threshold, min_supporting) for p in una["pairs"]]))
    return a, u


def best_gate_threshold(
    rows: pd.DataFrame, grid: np.ndarray, min_supporting: int
) -> tuple[float, float]:
    """Threshold with the best balanced accuracy (ties -> lower threshold) and that accuracy."""
    best_t, best_b = float(grid[0]), -1.0
    for t in grid:
        a, u = gate_rates(rows, float(t), min_supporting)
        if (a + u) / 2 > best_b + 1e-12:
            best_t, best_b = float(t), (a + u) / 2
    return best_t, best_b


def grouped_cv_gate(
    rows: pd.DataFrame, grid: np.ndarray, min_supporting: int, folds: int, seed: int
) -> dict[str, Any]:
    """Product-grouped k-fold gate estimate: tune on k-1 folds, measure on the held-out one."""
    items = sorted(rows["item"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(items)
    answer, refuse, thresholds = [], [], []
    for k in range(folds):
        held = set(items[k::folds])
        train, test = rows[~rows["item"].isin(held)], rows[rows["item"].isin(held)]
        t, _ = best_gate_threshold(train, grid, min_supporting)
        a, u = gate_rates(test, t, min_supporting)
        answer.append(a)
        refuse.append(u)
        thresholds.append(t)
    return {
        "fold_thresholds": thresholds,
        "answer_rate": float(np.mean(answer)),
        "refuse_rate": float(np.mean(refuse)),
    }
