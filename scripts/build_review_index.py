"""Phase 5 step 1: chunk the reviews and build the Chroma review index (resumable)."""

from __future__ import annotations

from catalogiq.logging_utils import get_logger
from catalogiq.rag.reviews_index import build_review_index, chunk_stats, load_chunks

logger = get_logger("build_review_index")


def main() -> None:
    chunks = load_chunks()
    logger.info("chunk stats: %s", chunk_stats(chunks))
    logger.info("collection size: %d", build_review_index(chunks))


if __name__ == "__main__":
    main()
