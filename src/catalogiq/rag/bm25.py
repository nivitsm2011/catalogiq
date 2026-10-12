"""A small in-house BM25 (Okapi) ranker; no extra dependency, fully inspectable."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = frozenset(
    "a an and are as at be by for from in is it of on or that the this to with your you our "
    "their its has have can will s".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stop words ("T-Shirt" -> ``["t", "shirt"]``)."""
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents.

    Args:
        documents: Raw document strings.
        k1: Term-frequency saturation.
        b: Length normalisation strength.
    """

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if len(documents) else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter[str] = Counter()
        for counts in self.tf:
            df.update(counts.keys())
        n = len(documents)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of ``query`` against every document."""
        out = np.zeros(len(self.tf))
        q_terms = [t for t in tokenize(query) if t in self.idf]
        for i, counts in enumerate(self.tf):
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0))
            for t in q_terms:
                f = counts.get(t, 0)
                if f:
                    out[i] += self.idf[t] * f * (self.k1 + 1) / (f + norm)
        return out
