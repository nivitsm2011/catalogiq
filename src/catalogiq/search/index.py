"""FAISS indexes over L2-normalised embeddings (cosine similarity = inner product).

Two kinds are supported:

* ``IndexFlatIP`` - exact brute force. Perfect recall; cost grows linearly with catalog size.
* ``IndexHNSWFlat`` - approximate graph index. Much faster queries at large scale for a small
  recall loss, at the price of build time, memory and tuning (``M``, ``efSearch``).
"""

from __future__ import annotations

import time
from typing import Any

import faiss
import numpy as np

from catalogiq.config import get_search_config
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation (float32, contiguous)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def build_flat(emb: np.ndarray) -> faiss.Index:
    """Exact inner-product index."""
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(l2_normalise(emb))
    return index


def build_hnsw(
    emb: np.ndarray,
    M: int | None = None,
    ef_construction: int | None = None,
    ef_search: int | None = None,
) -> faiss.Index:
    """Approximate HNSW inner-product index (parameters default to ``configs/search.yaml``)."""
    cfg = get_search_config()["index"]["hnsw"]
    index = faiss.IndexHNSWFlat(emb.shape[1], M or cfg["M"], faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction or cfg["ef_construction"]
    index.add(l2_normalise(emb))
    index.hnsw.efSearch = ef_search or cfg["ef_search"]
    return index


def build_index(emb: np.ndarray, kind: str = "flat") -> faiss.Index:
    """Build ``"flat"`` or ``"hnsw"``."""
    if kind == "flat":
        return build_flat(emb)
    if kind == "hnsw":
        return build_hnsw(emb)
    raise ValueError(f"unknown index kind: {kind}")


def compare_indexes(
    emb: np.ndarray,
    queries: np.ndarray,
    k: int = 10,
    ef_search_values: tuple[int, ...] = (16, 64, 256),
) -> list[dict[str, Any]]:
    """Speed and recall of HNSW (several ``efSearch``) against the exact flat index.

    Recall@k here means: of the exact top-k neighbours, how many HNSW also returns (averaged).

    Returns:
        One row per index configuration: build seconds, memory MB, single-query latency (ms,
        mean and p95), batch throughput (queries/s) and recall@k.
    """
    queries = l2_normalise(queries)
    rows: list[dict[str, Any]] = []

    def measure(
        name: str, index: faiss.Index, build_s: float, truth: np.ndarray | None
    ) -> np.ndarray:
        lat = []
        for q in queries[: min(200, len(queries))]:
            t0 = time.perf_counter()
            index.search(q[None], k)
            lat.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        _, ids = index.search(queries, k)
        batch_qps = len(queries) / (time.perf_counter() - t0)
        recall = (
            1.0
            if truth is None
            else float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ids, truth, strict=True)]))
        )
        rows.append(
            {
                "index": name,
                "build_s": round(build_s, 2),
                "single_ms_mean": round(float(np.mean(lat)), 3),
                "single_ms_p95": round(float(np.percentile(lat, 95)), 3),
                "batch_qps": round(batch_qps, 1),
                f"recall@{k}_vs_flat": round(recall, 4),
            }
        )
        return ids

    t0 = time.perf_counter()
    flat = build_flat(emb)
    truth = measure("IndexFlatIP (exact)", flat, time.perf_counter() - t0, None)
    for ef in ef_search_values:
        t0 = time.perf_counter()
        hnsw = build_hnsw(emb, ef_search=ef)
        measure(
            f"IndexHNSWFlat (M={get_search_config()['index']['hnsw']['M']}, efSearch={ef})",
            hnsw,
            time.perf_counter() - t0,
            truth,
        )
    rows[0]["memory_mb"] = round(emb.shape[0] * emb.shape[1] * 4 / 1e6, 1)
    return rows
