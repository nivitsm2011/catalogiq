"""Retrieval metrics.

Relevance for image search is "same articleType AND same colour as the query". Definitions
(k = cut-off):

* **Recall@k (hit rate)**: share of queries with at least one relevant result in the top k. With
  hundreds of relevant items per query, "fraction of all relevant items found" would be meaningless,
  so we use the hit-rate definition common in product search.
* **Precision@k**: average share of the top k that is relevant.
* **mAP@k**: mean over queries of average precision, where AP@k = sum_{i<=k}(P@i * rel_i) divided by
  min(k, number of relevant items in the gallery).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relevance_matrix(
    query_attrs: pd.DataFrame, gallery_attrs: pd.DataFrame, ranked: np.ndarray, keys: list[str]
) -> np.ndarray:
    """Boolean (n_queries, k) matrix: is each ranked gallery item relevant to its query?

    Args:
        query_attrs: Attributes of the queries (rows aligned with ``ranked``).
        gallery_attrs: Attributes of the gallery (``ranked`` holds positions into it).
        ranked: (n_queries, k) gallery positions ordered by decreasing similarity.
        keys: Attribute columns that must all match (e.g. ``["articleType", "baseColour"]``).
    """
    rel = np.ones(ranked.shape, dtype=bool)
    for key in keys:
        g = gallery_attrs[key].astype(str).to_numpy()[ranked]
        q = query_attrs[key].astype(str).to_numpy()[:, None]
        rel &= g == q
    return rel


def recall_at_k(rel: np.ndarray, k: int) -> float:
    """Hit rate: share of queries with at least one relevant result in the top k."""
    return float(rel[:, :k].any(axis=1).mean())


def precision_at_k(rel: np.ndarray, k: int) -> float:
    """Average fraction of the top k that is relevant."""
    return float(rel[:, :k].mean())


def average_precision_at_k(rel: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """mAP@k given the relevance matrix and the number of relevant gallery items per query."""
    rel_k = rel[:, :k].astype(float)
    precision_at_i = np.cumsum(rel_k, axis=1) / np.arange(1, k + 1)
    denom = np.minimum(k, np.maximum(n_relevant, 1))
    ap = (precision_at_i * rel_k).sum(axis=1) / denom
    ap[n_relevant == 0] = 0.0
    return float(ap.mean())


def count_relevant(
    query_attrs: pd.DataFrame, gallery_attrs: pd.DataFrame, keys: list[str]
) -> np.ndarray:
    """Number of gallery items that are relevant to each query (all ``keys`` equal)."""
    gallery_key = gallery_attrs[keys].astype(str).agg("|".join, axis=1)
    counts = gallery_key.value_counts()
    query_key = query_attrs[keys].astype(str).agg("|".join, axis=1)
    return query_key.map(counts).fillna(0).to_numpy().astype(int)
