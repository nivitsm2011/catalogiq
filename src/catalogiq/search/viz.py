"""Plotting helpers for the visual-search notebook (images come from the cached uint8 array)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from catalogiq.config import resolve_path
from catalogiq.search.looks import Look

GOOD, PARTIAL, BAD = "#1b7f3b", "#e0a33a", "#b3261e"


def _show(ax, image: np.ndarray, title: str, border: str | None = None) -> None:
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(border is not None)
        if border:
            spine.set_edgecolor(border)
            spine.set_linewidth(3)
    ax.set_title(title, fontsize=7, loc="left")


def _caption(row: pd.Series) -> str:
    """Two-line caption for a result tile."""
    return f"{row['articleType']}, {row['baseColour']}\n{row['gender']}  sim {row['score']:.2f}"


def _grade(query: pd.Series | None, row: pd.Series) -> str | None:
    """Border colour: green = same type and colour, amber = same type only, red = other type."""
    if query is None:
        return None
    if row["articleType"] == query["articleType"]:
        return GOOD if row["baseColour"] == query["baseColour"] else PARTIAL
    return BAD


def plot_query_grid(
    images: np.ndarray,
    catalog: pd.DataFrame,
    queries: Sequence[tuple[pd.Series, np.ndarray, pd.DataFrame]],
    path: str | None = None,
    title: str = "",
) -> plt.Figure:
    """One row per query: the query photo followed by its top results, graded by relevance.

    Args:
        images: uint8 image array aligned with ``catalog``.
        catalog: Catalog table (``id`` column) used to locate images.
        queries: ``(query_row, query_image, results_frame)`` triples.
        path: Optional output file (relative to the project root).
    """
    pos_of = {int(i): p for p, i in enumerate(catalog["id"])}
    n_res = max(len(r) for _, _, r in queries)
    fig, axes = plt.subplots(
        len(queries), n_res + 1, figsize=(1.7 * (n_res + 1), 2.25 * len(queries)), squeeze=False
    )
    for r, (q, qimg, res) in enumerate(queries):
        _show(
            axes[r, 0], qimg, f"QUERY\n{q['articleType']}, {q['baseColour']}\n{q['gender']}", None
        )
        axes[r, 0].add_patch(
            Rectangle((0, 0), 1, 1, transform=axes[r, 0].transAxes, fill=False, lw=3, ec="#2f6fb0")
        )
        for c in range(n_res):
            ax = axes[r, c + 1]
            if c >= len(res):
                ax.axis("off")
                continue
            row = res.iloc[c]
            _show(
                ax,
                images[pos_of[int(row["id"])]],
                _caption(row),
                _grade(q, row),
            )
    if title:
        fig.suptitle(title, fontsize=10, fontweight="bold")
    fig.text(
        0.01,
        0.005,
        "border: green = same type + colour, amber = same type, red = different type",
        fontsize=7,
    )
    plt.tight_layout()
    if path:
        fig.savefig(resolve_path(path), dpi=130, bbox_inches="tight")
    return fig


def plot_text_grid(
    images: np.ndarray,
    catalog: pd.DataFrame,
    results: Sequence[tuple[str, pd.DataFrame]],
    path: str | None = None,
) -> plt.Figure:
    """One row per text query with its top results (no relevance colouring)."""
    pos_of = {int(i): p for p, i in enumerate(catalog["id"])}
    n_res = max(len(r) for _, r in results)
    fig, axes = plt.subplots(
        len(results), n_res, figsize=(1.7 * n_res, 2.4 * len(results)), squeeze=False
    )
    for r, (text, res) in enumerate(results):
        for c in range(n_res):
            ax = axes[r, c]
            if c >= len(res):
                ax.axis("off")
                continue
            row = res.iloc[c]
            _show(
                ax,
                images[pos_of[int(row["id"])]],
                (f'"{text}"\n' if c == 0 else "\n") + _caption(row),
            )
    plt.tight_layout()
    if path:
        fig.savefig(resolve_path(path), dpi=130, bbox_inches="tight")
    return fig


def plot_looks(
    images: np.ndarray, catalog: pd.DataFrame, looks: Sequence[Look], path: str | None = None
) -> plt.Figure:
    """One row per look: the query item, then its recommendations grouped by role."""
    pos_of = {int(i): p for p, i in enumerate(catalog["id"])}
    n_cols = 1 + max(len(look.items) for look in looks)
    fig, axes = plt.subplots(
        len(looks), n_cols, figsize=(1.7 * n_cols, 2.5 * len(looks)), squeeze=False
    )
    for r, look in enumerate(looks):
        q = look.query
        _show(
            axes[r, 0],
            images[pos_of[int(q["id"])]],
            f"QUERY ({q['role']})\n{q['articleType']}, {q['baseColour']}\n"
            f"{q['gender']}, {q['usage']}",
            "#2f6fb0",
        )
        for c in range(1, n_cols):
            ax = axes[r, c]
            if c > len(look.items):
                ax.axis("off")
                continue
            row = look.items.iloc[c - 1]
            _show(
                ax,
                images[pos_of[int(row["id"])]],
                f"{row['role']}: {row['articleType']}\n{row['baseColour']}, {row['usage']}\n"
                f"score {row['score']:.2f}",
            )
    plt.tight_layout()
    if path:
        fig.savefig(resolve_path(path), dpi=130, bbox_inches="tight")
    return fig


def tsne_coordinates(
    emb: np.ndarray, n: int, seed: int, cache: str | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """2D t-SNE of a random subset of embeddings (PCA to 50 dims first); cached on disk.

    Returns:
        Subset row positions and their (n, 2) coordinates.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    path = resolve_path(cache) if cache else None
    if path is not None and path.exists():
        data = np.load(path)
        return data["pos"], data["xy"]
    rng = np.random.default_rng(seed)
    pos = np.sort(rng.choice(len(emb), min(n, len(emb)), replace=False))
    reduced = PCA(n_components=50, random_state=seed).fit_transform(emb[pos])
    xy = TSNE(n_components=2, perplexity=30, init="pca", random_state=seed).fit_transform(reduced)
    if path is not None:
        np.savez(path, pos=pos, xy=xy)
    return pos, xy


def plot_pairs(
    images: np.ndarray,
    catalog: pd.DataFrame,
    pairs: pd.DataFrame,
    n: int = 6,
    path: str | None = None,
    title: str = "",
) -> plt.Figure:
    """Show ``n`` pairs side by side with their similarity and pHash distance."""
    pos_of = {int(i): p for p, i in enumerate(catalog["id"])}
    shown = pairs.head(n)
    fig, axes = plt.subplots(len(shown), 2, figsize=(3.0, 2.4 * len(shown)), squeeze=False)
    for r, (_, p) in enumerate(shown.iterrows()):
        for c, key in enumerate(("id_a", "id_b")):
            row = catalog.iloc[pos_of[int(p[key])]]
            caption = f"{row['articleType']}, {row['baseColour']}"
            if c == 0:
                caption += f"\nsim {p['similarity']:.3f}, pHash {p['phash_distance']:.0f}"
            _show(axes[r, c], images[pos_of[int(p[key])]], caption)
    if title:
        fig.suptitle(title, fontsize=9, fontweight="bold")
    plt.tight_layout()
    if path:
        fig.savefig(resolve_path(path), dpi=130, bbox_inches="tight")
    return fig
