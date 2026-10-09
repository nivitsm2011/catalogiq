"""Search engine: similar products by image, products by text, with optional attribute filters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.search.embeddings import Encoder
from catalogiq.search.index import build_index, l2_normalise

FILTERABLE = (
    "gender",
    "masterCategory",
    "subCategory",
    "articleType",
    "baseColour",
    "usage",
    "season",
)
RESULT_COLUMNS = (
    "id",
    "score",
    "productDisplayName",
    "articleType",
    "baseColour",
    "gender",
    "usage",
    "season",
    "masterCategory",
    "subCategory",
)

Filters = Mapping[str, str | Sequence[str]]


class SearchEngine:
    """Nearest-neighbour search over cached CLIP image embeddings.

    Args:
        catalog: Metadata table, one row per embedding row (needs ``id`` and filter columns).
        image_emb: (N, D) image embeddings aligned with ``catalog``.
        encoder: Anything with ``encode_image`` / ``encode_text`` (the CLIP encoder). Needed
            only for ``search_by_image`` / ``search_by_text``; ``search_by_id`` uses cached vectors.
        index_kind: ``"flat"`` (exact, default) or ``"hnsw"`` (approximate).
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        image_emb: np.ndarray,
        encoder: Encoder | None = None,
        index_kind: str = "flat",
    ) -> None:
        if len(catalog) != len(image_emb):
            raise ValueError("catalog and image_emb must have the same number of rows")
        self.catalog = catalog.reset_index(drop=True)
        self.emb = l2_normalise(image_emb)
        self.encoder = encoder
        self.index_kind = index_kind
        self.index = build_index(self.emb, index_kind)
        self.id_to_pos = {int(i): p for p, i in enumerate(self.catalog["id"])}

    # ------------------------------------------------------------------ helpers
    def _mask(self, filters: Filters | None) -> np.ndarray | None:
        """Boolean mask of catalog rows passing all filters (``None`` = no filter)."""
        if not filters:
            return None
        mask = np.ones(len(self.catalog), dtype=bool)
        for key, wanted in filters.items():
            if key not in FILTERABLE:
                raise ValueError(f"cannot filter on {key!r}; choose from {FILTERABLE}")
            values = {wanted.lower()} if isinstance(wanted, str) else {w.lower() for w in wanted}
            mask &= self.catalog[key].astype(str).str.lower().isin(values).to_numpy()
        return mask

    def _frame(self, positions: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
        rows = self.catalog.iloc[positions]
        out = rows.reindex(columns=[c for c in RESULT_COLUMNS if c != "score"]).copy()
        out.insert(1, "score", np.asarray(scores, dtype=np.float32))
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        return out.reset_index(drop=True)

    def search_vector(
        self,
        vec: np.ndarray,
        k: int = 10,
        filters: Filters | None = None,
        exclude_ids: Sequence[int] = (),
    ) -> pd.DataFrame:
        """Top-``k`` catalog rows for a query vector.

        Without filters the FAISS index is used (``k`` plus the excluded rows is over-fetched).
        With filters the candidate set is restricted first and searched exactly, so a filter never
        returns fewer results than exist (post-filtering an approximate index can).
        """
        q = l2_normalise(np.asarray(vec, dtype=np.float32).reshape(1, -1))
        exclude = {self.id_to_pos[int(i)] for i in exclude_ids if int(i) in self.id_to_pos}
        mask = self._mask(filters)
        if mask is None:
            scores, ids = self.index.search(q, min(k + len(exclude), len(self.catalog)))
            pairs = [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
        else:
            pool = np.where(mask)[0]
            if len(pool) == 0:
                return self._frame(np.array([], dtype=int), np.array([]))
            sims = self.emb[pool] @ q[0]
            top = np.argsort(-sims)[: k + len(exclude)]
            pairs = [(int(pool[i]), float(sims[i])) for i in top]
        pairs = [(p, s) for p, s in pairs if p not in exclude][:k]
        if not pairs:
            return self._frame(np.array([], dtype=int), np.array([]))
        positions, scores_out = zip(*pairs, strict=True)
        return self._frame(np.array(positions), np.array(scores_out))

    # ------------------------------------------------------------------ public API
    def search_by_id(
        self, item_id: int, k: int = 10, filters: Filters | None = None, exclude_self: bool = True
    ) -> pd.DataFrame:
        """Products similar to a catalog item (uses its cached embedding)."""
        pos = self.id_to_pos[int(item_id)]
        return self.search_vector(
            self.emb[pos], k, filters, exclude_ids=[item_id] if exclude_self else ()
        )

    def search_by_image(
        self, image: Any, k: int = 10, filters: Filters | None = None
    ) -> pd.DataFrame:
        """Products visually similar to a query photo (PIL image or RGB array).

        Args:
            image: Query photo.
            k: Number of results.
            filters: Optional attribute filters, e.g. ``{"gender": "Men"}`` or
                ``{"baseColour": ["Black", "Grey"]}``.
        """
        if self.encoder is None:
            raise RuntimeError("an encoder is required for image queries")
        return self.search_vector(self.encoder.encode_image(image), k, filters)

    def search_by_text(
        self, query: str, k: int = 10, filters: Filters | None = None
    ) -> pd.DataFrame:
        """Products matching a text description, e.g. ``"black formal shoes for men"``."""
        if self.encoder is None:
            raise RuntimeError("an encoder is required for text queries")
        if not query.strip():
            raise ValueError("query text is empty")
        return self.search_vector(self.encoder.encode_text(query), k, filters)

    def search_matrix(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Exact top-``k`` positions for many query vectors at once (used by evaluation)."""
        q = l2_normalise(queries)
        out = np.empty((len(q), k), dtype=np.int64)
        for start in range(0, len(q), 512):
            sims = q[start : start + 512] @ self.emb.T
            top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
            order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
            out[start : start + 512] = np.take_along_axis(top, order, axis=1)
        return out
