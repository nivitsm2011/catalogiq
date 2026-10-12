"""Retrieval evaluation: BM25 vs vector vs hybrid (+ quality boost) on a small labelled set."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import yaml

from catalogiq.config import get_rag_config, resolve_path
from catalogiq.rag.retrieval import HybridRetriever


def load_queries() -> list[dict[str, str]]:
    """Queries with their relevance regexes from ``configs/retrieval_test_queries.yaml``."""
    path = resolve_path(get_rag_config()["paths"]["retrieval_queries"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))["queries"]


def relevance_mask(items: pd.DataFrame, pattern: str) -> np.ndarray:
    """True for listings whose title + features text matches the relevance regex."""
    rx = re.compile(pattern, re.IGNORECASE)
    text = (
        items["title"].astype(str)
        + " "
        + items["features"].map(lambda f: " ".join(map(str, f)) if f is not None else "")
    )
    return text.map(lambda t: bool(rx.search(t))).to_numpy()


def evaluate_retrieval(
    retriever: HybridRetriever, ks: tuple[int, ...] = (3, 5), boost: float | None = None
) -> pd.DataFrame:
    """Per-query, per-mode metrics: hit@k, precision@k, reciprocal rank, mean quality of top-k."""
    rows = []
    items = retriever.items
    for q in load_queries():
        rel = dict(zip(items["parent_asin"], relevance_mask(items, q["relevant"]), strict=True))
        n_rel = int(sum(rel.values()))
        for mode in ("bm25", "vector", "hybrid"):
            res = retriever.search(q["query"], k=max(ks), mode=mode, boost=boost)
            flags = [rel[a] for a in res["parent_asin"]]
            row = {
                "id": q["id"],
                "kind": q["kind"],
                "mode": mode,
                "n_relevant": n_rel,
                "mrr": next((1 / (i + 1) for i, f in enumerate(flags) if f), 0.0),
                "quality_top": (
                    float(res["quality_score"].head(max(ks)).mean()) if len(res) else 0.0
                ),
            }
            for k in ks:
                row[f"hit@{k}"] = float(any(flags[:k]))
                row[f"p@{k}"] = float(np.mean(flags[:k])) if flags[:k] else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per (kind, mode) and overall."""
    cols = [c for c in df.columns if c.startswith(("hit@", "p@")) or c in ("mrr", "quality_top")]
    by_kind = df.groupby(["kind", "mode"])[cols].mean()
    overall = df.groupby("mode")[cols].mean()
    overall.index = pd.MultiIndex.from_product([["ALL"], overall.index])
    return pd.concat([by_kind, overall])
