"""CLIP zero-shot text-overlay score.

A blob heuristic for text and watermarks (``image_checks.text_row_score``) found almost nothing at a
3% false-alarm rate, while CLIP, which can read text even in 60x80 thumbnails, found about half. The
score is ``100 * (cos(image, "text overlay") - cos(image, "clean photo"))``.
"""

from __future__ import annotations

import numpy as np

from catalogiq.config import get_quality_config
from catalogiq.search.embeddings import ClipEncoder


def prompt_embeddings(encoder: ClipEncoder) -> np.ndarray:
    """(2, 512) embeddings of the text-overlay and the clean-photo prompts from the config."""
    return encoder.encode_texts(list(get_quality_config()["image"]["text_prompts"]))


def text_margin(image_embeddings: np.ndarray, prompts: np.ndarray) -> np.ndarray:
    """Zero-shot text-overlay score per image; higher means more likely to carry overlaid text."""
    sims = 100.0 * np.atleast_2d(image_embeddings) @ prompts.T
    return sims[:, 0] - sims[:, 1]
