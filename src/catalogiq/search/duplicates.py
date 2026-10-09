"""Near-duplicate listing detection: CLIP similarity threshold, confirmed by a perceptual hash.

Two photos are flagged as duplicates when (a) their CLIP embeddings have cosine similarity at or
above ``threshold`` and (b) optionally their 64-bit perceptual hashes (pHash) differ by at most
``max_phash_distance`` bits. The embedding finds "same product" even when the photo was re-shot or
re-cropped; the hash is a cheap, independent check that removes look-alike but different products
(for example two black loafers) that CLIP also scores very high.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_search_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.search.index import build_flat, l2_normalise

logger = get_logger(__name__)


def hamming_hex(a: str | None, b: str | None) -> float:
    """Hamming distance between two hex hash strings (NaN if either is missing)."""
    if not isinstance(a, str) or not isinstance(b, str):
        return float("nan")
    return float(bin(int(a, 16) ^ int(b, 16)).count("1"))


def load_phash(catalog_ids: Sequence[int]) -> pd.Series:
    """pHash strings (from the Phase 1 audit) indexed by catalog id."""
    audit = pd.read_parquet(resolve_path(get_config()["data"]["fashion"]["audit_parquet"]))
    phash = audit.drop_duplicates("id").set_index("id")["phash"]
    return phash.reindex(list(catalog_ids))


class DuplicateFinder:
    """Find near-duplicate listings in a catalog.

    Args:
        catalog: Metadata with an ``id`` column, rows aligned with ``emb``.
        emb: (N, D) embeddings (any scale; normalised internally).
        phash_by_id: Optional ``Series`` of hex pHashes indexed by id (for pHash confirmation).
        candidate_k: Neighbours examined per item.
        min_similarity: Candidate pairs below this cosine similarity are never considered.
    """

    def __init__(
        self,
        catalog: pd.DataFrame,
        emb: np.ndarray,
        phash_by_id: pd.Series | None = None,
        candidate_k: int | None = None,
        min_similarity: float = 0.75,
    ) -> None:
        self.catalog = catalog.reset_index(drop=True)
        self.emb = l2_normalise(emb)
        self.phash_by_id = phash_by_id
        self.k = candidate_k or get_search_config()["duplicates"]["candidate_k"]
        self.min_similarity = min_similarity
        self._pairs: pd.DataFrame | None = None

    def candidate_pairs(self) -> pd.DataFrame:
        """All item pairs that are among each other's ``k`` nearest neighbours (cached).

        Returns:
            Columns ``id_a``, ``id_b`` (a before b in the catalog), ``similarity`` and
            ``phash_distance``.
        """
        if self._pairs is not None:
            return self._pairs
        index = build_flat(self.emb)
        sims, nbrs = index.search(self.emb, self.k + 1)
        rows = np.repeat(np.arange(len(self.emb)), self.k + 1)
        pairs = pd.DataFrame({"i": rows, "j": nbrs.ravel(), "similarity": sims.ravel()})
        pairs = pairs[(pairs["i"] != pairs["j"]) & (pairs["j"] >= 0)]
        pairs = pairs[pairs["similarity"] >= self.min_similarity]
        lo, hi = np.minimum(pairs["i"], pairs["j"]), np.maximum(pairs["i"], pairs["j"])
        pairs = pairs.assign(i=lo, j=hi).drop_duplicates(["i", "j"]).reset_index(drop=True)
        ids = self.catalog["id"].to_numpy()
        pairs["id_a"], pairs["id_b"] = ids[pairs["i"]], ids[pairs["j"]]
        if self.phash_by_id is not None:
            ha, hb = (
                self.phash_by_id.loc[pairs["id_a"]].to_numpy(),
                self.phash_by_id.loc[pairs["id_b"]].to_numpy(),
            )
            pairs["phash_distance"] = [hamming_hex(a, b) for a, b in zip(ha, hb, strict=True)]
        else:
            pairs["phash_distance"] = np.nan
        self._pairs = pairs[["id_a", "id_b", "similarity", "phash_distance"]]
        logger.info(
            "%d candidate pairs with similarity >= %.2f", len(self._pairs), self.min_similarity
        )
        return self._pairs

    def find_duplicates(
        self,
        threshold: float,
        max_phash_distance: int | None = None,
        require_phash: bool = True,
    ) -> pd.DataFrame:
        """Pairs flagged as duplicates.

        Args:
            threshold: Minimum CLIP cosine similarity.
            max_phash_distance: Maximum pHash Hamming distance (default from config).
            require_phash: If False, flag on similarity alone.

        Returns:
            The flagged pairs, most similar first.
        """
        pairs = self.candidate_pairs()
        keep = pairs["similarity"] >= threshold
        if require_phash:
            limit = (
                max_phash_distance
                if max_phash_distance is not None
                else get_search_config()["duplicates"]["phash_max_distance"]
            )
            keep &= pairs["phash_distance"] <= limit
        return pairs[keep].sort_values("similarity", ascending=False).reset_index(drop=True)


def duplicate_groups(pairs: pd.DataFrame) -> dict[int, int]:
    """Cluster flagged pairs into groups (connected components); returns ``{id: group_id}``."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(pairs["id_a"], pairs["id_b"], strict=True):
        parent[find(int(a))] = find(int(b))
    roots: dict[int, int] = {}
    return {x: roots.setdefault(find(x), len(roots)) for x in list(parent)}


def evaluate_thresholds(
    labelled: pd.DataFrame, thresholds: Sequence[float], max_phash_distance: int
) -> pd.DataFrame:
    """Precision/recall/F1 of every threshold on hand-labelled pairs, with and without pHash.

    Args:
        labelled: Columns ``similarity``, ``phash_distance``, ``label`` (1 duplicate, 0 not) and
            optionally ``weight`` (sampling weight so a stratified sample estimates the whole
            candidate population).
        thresholds: Similarity thresholds to try.
        max_phash_distance: pHash limit for the "with pHash" variant.
    """
    w = labelled["weight"].to_numpy() if "weight" in labelled else np.ones(len(labelled))
    y = labelled["label"].to_numpy().astype(bool)
    rows = []
    for use_hash in (False, True):
        for t in thresholds:
            pred = (labelled["similarity"] >= t).to_numpy()
            if use_hash:
                pred = pred & (labelled["phash_distance"] <= max_phash_distance).to_numpy()
            tp, fp, fn = (w * (pred & y)).sum(), (w * (pred & ~y)).sum(), (w * (~pred & y)).sum()
            precision = tp / (tp + fp) if tp + fp else float("nan")
            recall = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
            rows.append(
                {
                    "method": "similarity + pHash" if use_hash else "similarity only",
                    "threshold": round(float(t), 4),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "pairs_flagged_in_sample": int(pred.sum()),
                }
            )
    return pd.DataFrame(rows)
