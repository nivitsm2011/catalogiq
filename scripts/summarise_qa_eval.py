"""Collect every Phase 5 headline number into reports/phase5/metrics.json (no new model calls).

Usage: python scripts/summarise_qa_eval.py   (after scripts/evaluate_qa.py all)
"""

from __future__ import annotations

import json

import mlflow
import numpy as np
import pandas as pd

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.rag.eval_qa import (
    answer_metrics,
    ground_truth,
    load_questions,
    mean_ci,
    review_texts,
)
from catalogiq.tracking import start_run


def _jsonl(path: object) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]  # type: ignore[attr-defined]


def main() -> None:
    """Write metrics.json."""
    cfg = get_qa_config()
    out = resolve_path(cfg["paths"]["eval_dir"])
    reviews = review_texts()
    gt = ground_truth(load_questions(), reviews).set_index("id", drop=False)
    text_of = dict(zip(reviews["review_id"], reviews["full"], strict=True))
    answers = pd.DataFrame(_jsonl(out / "answers.jsonl"))
    faith = pd.DataFrame(_jsonl(out / "faithfulness.jsonl"))
    m = answer_metrics(answers, faith, gt, text_of)
    ret = pd.read_pickle(out / "retrieval.pkl")
    rec = {}
    for v, g in ret[ret["kind"] == "answerable"].groupby("variant"):
        r5, lo, hi = mean_ci(g["recall5"].to_numpy())
        rec[v] = {
            "recall5": r5,
            "ci": [lo, hi],
            "mrr": float(g["mrr"].mean()),
            "latency_ms": float(ret[ret["variant"] == v]["latency_ms"].mean()),
        }
    # judge vs human on the 15 hand-checked answers
    key = pd.read_csv(out / "qa_handcheck_key.csv")
    hum = pd.read_csv(out / "qa_handcheck_ratings.csv")
    d = key.merge(hum, on="item")
    d["human"] = d["faithful"].eq("yes")
    hc = {
        "n": len(d),
        "agreement": float((d["judge_faithful"] == d["human"]).mean()),
        "human_faithful_rate": float(d["human"].mean()),
        "judge_flagged_overturned": int(((~d["judge_faithful"]) & d["human"]).sum()),
        "judge_flagged": int((~d["judge_faithful"]).sum()),
        "judge_passed_human_rejected": int((d["judge_faithful"] & ~d["human"]).sum()),
    }
    # answered-question accuracy of the gate at the end-to-end level
    a = answers.copy()
    a["refused"] = a["output"].map(lambda o: o["refused"])
    conf = a[~a["refused"]]["output"].map(lambda o: o["confidence"])
    summ = json.loads((out / "summaries.json").read_text(encoding="utf-8"))
    src = pd.Series([p["source"] for p in summ["products"].values()]).value_counts().to_dict()
    labels = pd.read_csv(out / "aspect_labels.csv")
    res = {
        "end_to_end": m,
        "retrieval": rec,
        "handcheck": hc,
        "mean_confidence_answered": float(np.mean(conf)),
        "summary_sources": src,
        "summary_method": summ["method"],
        "gate": json.loads((out / "gate.json").read_text()),
        "aspect": json.loads(resolve_path(cfg["paths"]["aspect_results"]).read_text("utf-8")),
        "label_positives": {c: int(labels[c].sum()) for c in labels.columns if c != "chunk_id"},
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    flat = {f"e2e_{k}": v for k, v in m.items() if isinstance(v, (int, float))}
    flat |= {f"recall5_{v}": r["recall5"] for v, r in rec.items()}
    flat |= {f"mrr_{v}": r["mrr"] for v, r in rec.items()}
    flat |= {f"handcheck_{k}": v for k, v in hc.items()}
    flat["gate_min_similarity"] = res["gate"]["min_similarity"]
    with start_run("phase5_review_qa", {"model": "llama3.2:3b", "aspect_method": summ["method"]}):
        mlflow.log_metrics({k: float(v) for k, v in flat.items()})
    print(
        json.dumps(
            {k: res[k] for k in ("end_to_end", "retrieval", "handcheck", "summary_sources")},
            indent=1,
            default=float,
        )
    )


if __name__ == "__main__":
    main()
