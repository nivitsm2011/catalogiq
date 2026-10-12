"""Hybrid retrieval: BM25 + vector, rank fusion, listing-quality boost, category filter."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from catalogiq.config import get_rag_config
from catalogiq.logging_utils import get_logger
from catalogiq.rag.bm25 import BM25
from catalogiq.rag.knowledge_base import embed_texts, get_client, load_listings

logger = get_logger(__name__)

Mode = Literal["hybrid", "bm25", "vector"]


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal rank fusion: ``score(d) = sum over rankers of 1 / (k + rank)``, rank from 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Searches the listing knowledge base.

    Args:
        items: Listing table from ``load_listings`` (must match what is indexed in Chroma).
        collection: Chroma listings collection. Opened from the config when omitted.
        style_collection: Chroma style-guide collection.
    """

    def __init__(
        self,
        items: pd.DataFrame | None = None,
        collection: Any = None,
        style_collection: Any = None,
    ) -> None:
        cfg = get_rag_config()
        self.cfg = cfg["retrieval"]
        self.items = (items if items is not None else load_listings()).reset_index(drop=True)
        if collection is None:
            client = get_client()
            collection = client.get_collection(cfg["paths"]["collection_listings"])
            style_collection = client.get_collection(cfg["paths"]["collection_style"])
        self.collection = collection
        self.style_collection = style_collection
        self.bm25 = BM25(self.items["doc"].tolist(), **self.cfg["bm25"])
        self._ids = self.items["parent_asin"].tolist()
        self._quality = dict(zip(self._ids, self.items["quality_score"], strict=True))
        self._group = self.items["group"].to_numpy()

    def _filter_group(self, group: str | None) -> str | None:
        """Use the category filter only when the category has enough listings."""
        if group and int((self._group == group).sum()) >= self.cfg["min_group_size"]:
            return group
        return None

    def bm25_ranking(self, query: str, k: int, group: str | None = None) -> list[str]:
        """Asins ranked by BM25 (best first), optionally limited to one category group."""
        scores = self.bm25.scores(query)
        if group:
            scores = np.where(self._group == group, scores, -1.0)
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._ids[i] for i in order if scores[i] > 0]

    def vector_ranking(self, query: str, k: int, group: str | None = None) -> list[str]:
        """Asins ranked by cosine similarity of MiniLM embeddings (best first)."""
        where = {"group": group} if group else None
        res = self.collection.query(
            query_embeddings=embed_texts([query]).tolist(), n_results=k, where=where
        )
        return list(res["ids"][0])

    def search(
        self,
        query: str,
        k: int = 5,
        group: str | None = None,
        mode: Mode = "hybrid",
        boost: float | None = None,
    ) -> pd.DataFrame:
        """Top-``k`` listings for ``query``.

        Args:
            query: Free-text query.
            k: Number of listings returned.
            group: Category group filter (ignored when the group is too small).
            mode: ``hybrid`` (RRF of both rankers), ``bm25`` or ``vector``.
            boost: Quality-boost weight; defaults to the config value. 0 switches it off.

        Returns:
            Listing rows with ``score`` (higher is better), best first.
        """
        group = self._filter_group(group)
        n = self.cfg["candidate_k"]
        boost = self.cfg["quality_boost"] if boost is None else boost
        rankings: list[list[str]] = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self.bm25_ranking(query, n, group))
        if mode in ("hybrid", "vector"):
            rankings.append(self.vector_ranking(query, n, group))
        fused = rrf_fuse(rankings, self.cfg["rrf_k"])
        scored = {a: s * (1.0 + boost * self._quality[a]) for a, s in fused.items()}
        top = sorted(scored, key=scored.get, reverse=True)[:k]  # type: ignore[arg-type]
        out = self.items.set_index("parent_asin").loc[top].reset_index()
        out["score"] = [scored[a] for a in top]
        return out

    def style_rules(self, group: str) -> str:
        """Global + category style-guide chunks, joined as prompt text."""
        ids = ["style_all", f"style_{group}"]
        res = self.style_collection.get(ids=ids)
        by_id = dict(zip(res["ids"], res["documents"], strict=True))
        return "\n".join(by_id[i] for i in ids if i in by_id)
