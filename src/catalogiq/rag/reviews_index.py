"""Review index: chunk Amazon reviews, store embeddings + metadata in a Chroma collection.

Short reviews stay whole (one chunk). Long reviews are split into overlapping sentence windows so
that one chunk talks about one thing. A chunk keeps its review id, so answers can cite ``[R123]``.
The build is resumable: embeddings are added block by block and ids already stored are skipped.
"""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_qa_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.rag.knowledge_base import embed_texts, get_client

logger = get_logger(__name__)

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Rough sentence split on end punctuation followed by whitespace."""
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def chunk_review(
    title: str, text: str, short_max: int, window: int, stride: int, max_chars: int, prepend: bool
) -> list[str]:
    """Chunks of one review: whole if short, else sentence windows (overlap = window - stride)."""
    title, text = (title or "").strip(), (text or "").strip()
    full = f"{title}. {text}" if prepend and title else text
    if len(text) <= short_max:
        return [full[:max_chars]]
    sents = split_sentences(text)
    chunks: list[str] = []
    for start in range(0, len(sents), stride):
        piece = " ".join(sents[start : start + window])[:max_chars]
        if piece:
            chunks.append(piece)
        if start + window >= len(sents):
            break
    if prepend and title and chunks:
        chunks[0] = f"{title}. {chunks[0]}"[:max_chars]
    return chunks


def build_chunks(reviews: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per chunk with review metadata (``review_id`` = row in the reviews parquet)."""
    cfg = get_qa_config()["chunking"]
    if reviews is None:
        reviews = pd.read_parquet(resolve_path(get_config()["data"]["reviews"]["reviews_parquet"]))
    rows: list[dict[str, Any]] = []
    for rid, r in enumerate(reviews.itertuples(index=False)):
        pieces = chunk_review(
            r.title,
            r.text,
            cfg["short_max_chars"],
            cfg["window_sentences"],
            cfg["stride_sentences"],
            cfg["max_chunk_chars"],
            cfg["prepend_title"],
        )
        for k, piece in enumerate(pieces):
            rows.append(
                {
                    "chunk_id": f"R{rid}_{k}",
                    "review_id": f"R{rid}",
                    "item_id": r.parent_asin,
                    "rating": float(r.rating),
                    "verified_purchase": bool(r.verified_purchase),
                    "helpful_vote": int(r.helpful_vote),
                    "date": r.timestamp.strftime("%Y-%m-%d"),
                    "chunk_idx": k,
                    "n_chunks": len(pieces),
                    "text": piece,
                }
            )
    return pd.DataFrame(rows)


def load_chunks(rebuild: bool = False) -> pd.DataFrame:
    """Chunk table from its parquet (built on first use)."""
    path = resolve_path(get_qa_config()["paths"]["chunks_parquet"])
    if path.exists() and not rebuild:
        return pd.read_parquet(path)
    chunks = build_chunks()
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks.to_parquet(path)
    logger.info("Chunked reviews -> %d chunks (%s)", len(chunks), path)
    return chunks


def build_review_index(chunks: pd.DataFrame | None = None) -> int:
    """Embed every chunk and store it in Chroma (resumable); returns the collection size."""
    cfg = get_qa_config()
    chunks = load_chunks() if chunks is None else chunks
    col = get_client().get_or_create_collection(
        cfg["paths"]["collection"], metadata={"hnsw:space": "cosine"}
    )
    done = set()
    existing = col.count()
    if existing:
        for start in range(0, existing, 20000):
            done.update(col.get(include=[], limit=20000, offset=start)["ids"])
    todo = chunks[~chunks["chunk_id"].isin(done)]
    logger.info("%d chunks total, %d already indexed, %d to do", len(chunks), len(done), len(todo))
    block = cfg["chunking"]["embed_block"]
    sub = 1000  # Chroma add() batch size
    t0 = time.perf_counter()
    for b in range(0, len(todo), block):
        part = todo.iloc[b : b + block]
        emb = embed_texts(part["text"].tolist())
        for s in range(0, len(part), sub):
            piece = part.iloc[s : s + sub]
            col.add(
                ids=piece["chunk_id"].tolist(),
                embeddings=emb[s : s + sub].tolist(),
                documents=piece["text"].tolist(),
                metadatas=[
                    {
                        "review_id": r.review_id,
                        "item_id": r.item_id,
                        "rating": r.rating,
                        "verified_purchase": r.verified_purchase,
                        "helpful_vote": r.helpful_vote,
                        "date": r.date,
                        "chunk_idx": r.chunk_idx,
                    }
                    for r in piece.itertuples()
                ],
            )
        rate = (b + len(part)) / (time.perf_counter() - t0)
        logger.info("indexed %d/%d chunks (%.0f chunks/s)", b + len(part), len(todo), rate)
    return col.count()


def chunk_stats(chunks: pd.DataFrame) -> dict[str, float]:
    """Headline numbers for the notebook."""
    per_review = chunks.groupby("review_id")["chunk_idx"].count()
    return {
        "reviews": int(per_review.size),
        "chunks": int(len(chunks)),
        "split_reviews": int((per_review > 1).sum()),
        "mean_chars": float(chunks["text"].str.len().mean()),
        "max_chars": int(chunks["text"].str.len().max()),
        "max_chunks_per_review": int(per_review.max()),
        "median_chunks_per_item": float(np.median(chunks.groupby("item_id").size())),
    }
