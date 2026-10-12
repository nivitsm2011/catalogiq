"""Knowledge base: Amazon listings (one document each) and style-guide chunks in ChromaDB.

One listing = one document (title + features + description) with metadata (brand, category group,
rating, listing quality score, exemplar flag). Vectors are produced here with sentence-transformers
and handed to Chroma, so Chroma never downloads a model of its own.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.config import get_config, get_rag_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.rag.categories import group_from_title
from catalogiq.rag.style_guide import style_chunks
from catalogiq.utils import configure_cache_dirs

logger = get_logger(__name__)


def listing_text(row: pd.Series, max_chars: int) -> str:
    """Document text of one listing: title, then features, then description."""
    feats = [str(f).strip() for f in (row["features"] if row["features"] is not None else [])]
    parts = [str(row["title"]).strip()]
    if feats:
        parts.append("Features: " + "; ".join(feats))
    desc = str(row["description"] or "").strip()
    if desc:
        parts.append("Description: " + desc)
    return ". ".join(parts)[:max_chars]


def load_listings() -> pd.DataFrame:
    """Amazon items with ``group`` and ``doc`` columns added (the knowledge-base corpus)."""
    cfg = get_rag_config()
    items = pd.read_parquet(resolve_path(get_config()["data"]["reviews"]["meta_parquet"]))
    items = items.reset_index(drop=True)
    items["group"] = items["title"].map(group_from_title)
    items["doc"] = items.apply(listing_text, axis=1, max_chars=cfg["embedder"]["max_chars"])
    return items


@lru_cache(maxsize=1)
def get_embedder() -> Any:
    """Sentence-transformers model named in the config (cached under ``models/cache``)."""
    configure_cache_dirs()
    from sentence_transformers import SentenceTransformer

    name = get_rag_config()["embedder"]["name"]
    logger.info("Loading embedder %s", name)
    return SentenceTransformer(name, device="cpu")


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, shape (n, dim)."""
    bs = get_rag_config()["embedder"]["batch_size"]
    vecs = get_embedder().encode(
        texts, batch_size=bs, normalize_embeddings=True, show_progress_bar=False
    )
    return np.asarray(vecs, dtype="float32")


def get_client() -> Any:
    """Persistent ChromaDB client at ``paths.chroma_dir``."""
    import chromadb

    path = resolve_path(get_rag_config()["paths"]["chroma_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def build_knowledge_base(rebuild: bool = True) -> dict[str, int]:
    """Embed all listings and style chunks and store them in Chroma.

    Returns:
        Counts per collection.
    """
    cfg = get_rag_config()
    client = get_client()
    items = load_listings()
    names = (cfg["paths"]["collection_listings"], cfg["paths"]["collection_style"])
    if rebuild:
        for name in names:
            try:
                client.delete_collection(name)
            except Exception:  # noqa: BLE001 - collection may not exist yet
                pass
    meta = {"hnsw:space": "cosine"}
    col = client.get_or_create_collection(names[0], metadata=meta)
    emb = embed_texts(items["doc"].tolist())
    metadatas = [
        {
            "asin": r.parent_asin,
            "title": r.title,
            "brand": str(r.store or ""),
            "group": r.group,
            "rating": float(r.average_rating),
            "rating_number": int(r.rating_number),
            "quality_score": float(r.quality_score),
            "is_exemplar": bool(r.is_exemplar),
        }
        for r in items.itertuples()
    ]
    col.add(
        ids=items["parent_asin"].tolist(),
        embeddings=emb.tolist(),
        documents=items["doc"].tolist(),
        metadatas=metadatas,
    )
    style = style_chunks()
    scol = client.get_or_create_collection(names[1], metadata=meta)
    scol.add(
        ids=[c["id"] for c in style],
        embeddings=embed_texts([c["text"] for c in style]).tolist(),
        documents=[c["text"] for c in style],
        metadatas=[{"group": c["group"]} for c in style],
    )
    logger.info("Indexed %d listings and %d style chunks", len(items), len(style))
    return {names[0]: len(items), names[1]: len(style)}
