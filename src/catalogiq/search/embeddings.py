"""OpenCLIP embeddings for catalog images and attribute text, cached to disk.

Catalog photos are tiny (about 60x80 px) with white backgrounds. OpenCLIP's default preprocessing
resizes the short side and centre-crops, which would cut off heads and shoes, so we pad to a square
with white first and then upsample to 224x224 (bicubic, done with torch ops on whole batches).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from catalogiq.config import get_config, get_search_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.utils import configure_cache_dirs

logger = get_logger(__name__)


class Encoder(Protocol):
    """Anything that can embed images and text into the same normalised space."""

    def encode_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """One image -> L2-normalised vector."""

    def encode_text(self, text: str) -> np.ndarray:
        """One string -> L2-normalised vector."""


def _title(value: Any) -> str | None:
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)


def build_text(row: pd.Series, template: str | None = None) -> str:
    """Attribute sentence for one catalog row, e.g. "a product photo of a black shirts for men".

    Missing attributes are dropped instead of printed as "nan". The free-text product name is
    deliberately NOT used: the text must describe the attributes the classifier also predicts.
    """
    colour, atype = _title(row.get("baseColour")), _title(row.get("articleType"))
    gender, usage, season = (_title(row.get(k)) for k in ("gender", "usage", "season"))
    parts = ["a product photo of a"]
    if colour and colour != "Other":
        parts.append(colour.lower())
    parts.append(atype.lower() if atype else "fashion item")
    if gender:
        parts.append(f"for {gender.lower()}")
    sentence = " ".join(parts)
    extras = []
    if usage and usage != "Other":
        extras.append(f"{usage.lower()} style")
    if season:
        extras.append(f"{season.lower()} season")
    if extras:
        sentence += ", " + ", ".join(extras)
    return sentence


def preprocess_u8(images_u8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> CLIP-ready (N, 3, 224, 224): white-pad to square, resize, normalise."""
    scfg = get_search_config()["clip"]
    x = torch.from_numpy(np.array(images_u8)).permute(0, 3, 1, 2).float() / 255.0
    _, _, h, w = x.shape
    side = max(h, w)
    pad_l, pad_t = (side - w) // 2, (side - h) // 2
    x = F.pad(x, (pad_l, side - w - pad_l, pad_t, side - h - pad_t), value=scfg["pad_value"])
    size = scfg["image_size"]
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    mean = torch.tensor(scfg["mean"]).view(1, 3, 1, 1)
    std = torch.tensor(scfg["std"]).view(1, 3, 1, 1)
    return (x - mean) / std


class ClipEncoder:
    """OpenCLIP ViT-B-32 (laion2b_s34b_b79k) wrapper returning L2-normalised float32 vectors."""

    def __init__(self) -> None:
        configure_cache_dirs()
        import open_clip

        cfg = get_config()["models"]["clip"]
        self.model, _, _ = open_clip.create_model_and_transforms(
            cfg["name"], pretrained=cfg["pretrained"]
        )
        self.tokenizer = open_clip.get_tokenizer(cfg["name"])
        self.model.eval()

    @torch.no_grad()
    def encode_images_u8(self, images_u8: np.ndarray) -> np.ndarray:
        """Batch of uint8 images -> (N, 512) normalised embeddings."""
        feats = self.model.encode_image(preprocess_u8(images_u8))
        return F.normalize(feats, dim=-1).numpy().astype(np.float32)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Strings -> (N, 512) normalised embeddings."""
        feats = self.model.encode_text(self.tokenizer(texts))
        return F.normalize(feats, dim=-1).numpy().astype(np.float32)

    def encode_image(self, image: Image.Image | np.ndarray) -> np.ndarray:
        """One PIL image or array -> normalised vector."""
        arr = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
        return self.encode_images_u8(arr[None])[0]

    def encode_text(self, text: str) -> np.ndarray:
        """One string -> normalised vector."""
        return self.encode_texts([text])[0]


def build_catalog_table() -> pd.DataFrame:
    """Catalog metadata aligned with the cached image array (one row per cleaned image)."""
    from catalogiq.vision.data import build_image_cache

    cfg = get_config()
    _, index = build_image_cache()
    clean = pd.read_parquet(resolve_path(cfg["data"]["fashion"]["clean_parquet"]))
    keep = ["id", "productDisplayName", "baseColour_raw"]
    table = index.merge(clean[keep], on="id", how="left")
    table["position"] = np.arange(len(table))
    return table


def embed_catalog(force: bool = False) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Embed every catalog image and its attribute sentence; resumable and cached.

    Returns:
        Image embeddings (N, 512), text embeddings (N, 512) and the catalog table (row-aligned).
    """
    from catalogiq.vision.data import build_image_cache

    scfg = get_search_config()
    paths = {k: resolve_path(v) for k, v in scfg["paths"].items()}
    if not force and all(paths[k].exists() for k in ("image_emb", "text_emb", "id_map", "catalog")):
        logger.info("loading cached CLIP embeddings from %s", paths["image_emb"].parent)
        return (
            np.load(paths["image_emb"]),
            np.load(paths["text_emb"]),
            pd.read_parquet(paths["catalog"]),
        )

    images, _ = build_image_cache()
    catalog = build_catalog_table()
    encoder = ClipEncoder()
    bs, chunk_batches = scfg["clip"]["batch_size"], scfg["clip"]["chunk_batches"]
    chunk_dir = paths["chunks_dir"]
    chunk_dir.mkdir(parents=True, exist_ok=True)

    # --- images (chunked so an interruption loses at most chunk_batches batches) ---
    n = len(catalog)
    chunk_rows = bs * chunk_batches
    starts = list(range(0, n, chunk_rows))
    start_time = time.perf_counter()
    for ci, s0 in enumerate(tqdm(starts, desc="CLIP image chunks", mininterval=30)):
        f = chunk_dir / f"img_{s0:06d}.npy"
        if f.exists() and not force:
            continue
        out = [
            encoder.encode_images_u8(np.asarray(images[b : min(b + bs, n)]))
            for b in range(s0, min(s0 + chunk_rows, n), bs)
        ]
        np.save(f, np.concatenate(out))
        done = min((ci + 1) * chunk_rows, n)
        rate = done / (time.perf_counter() - start_time)
        logger.info("images %d/%d (%.1f img/s since start)", done, n, rate)
    image_emb = np.concatenate([np.load(chunk_dir / f"img_{s0:06d}.npy") for s0 in starts])

    # --- text ---
    texts = [build_text(r, scfg["text_template"]) for _, r in catalog.iterrows()]
    unique = sorted(set(texts))
    lookup = {t: i for i, t in enumerate(unique)}
    uniq_emb = np.concatenate(
        [encoder.encode_texts(unique[b : b + 256]) for b in range(0, len(unique), 256)]
    )
    text_emb = uniq_emb[[lookup[t] for t in texts]]
    catalog["text"] = texts

    paths["image_emb"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["image_emb"], image_emb)
    np.save(paths["text_emb"], text_emb)
    np.save(paths["id_map"], catalog["id"].to_numpy())
    catalog.to_parquet(paths["catalog"], index=False)
    logger.info(
        "saved embeddings: images %s, text %s (%d unique sentences)",
        image_emb.shape,
        text_emb.shape,
        len(unique),
    )
    return image_emb, text_emb, catalog


def classifier_embeddings(force: bool = False) -> np.ndarray:
    """Pooled 960-d features of the fine-tuned Phase 2 classifier for every catalog image.

    Used as the retrieval baseline ("what if we searched with our own classifier's features").
    L2-normalised so cosine similarity applies.
    """
    from catalogiq.vision.data import build_image_cache, normalise, resize, to_tensor
    from catalogiq.vision.model import MultiHeadNet

    path = resolve_path(get_search_config()["paths"]["classifier_emb"])
    if path.exists() and not force:
        return np.load(path)
    vcfg = get_vision_config()
    model_dir = resolve_path(vcfg["inference"]["model_dir"])
    import json

    labels = json.loads((model_dir / vcfg["inference"]["labels_file"]).read_text(encoding="utf-8"))
    net = MultiHeadNet({t: len(c) for t, c in labels.items()}, pretrained=False)
    net.load_state_dict(torch.load(Path(model_dir) / "best_full.pt"))
    net.eval()
    images, _ = build_image_cache()
    hw = tuple(vcfg["input_hw"])
    out = []
    with torch.no_grad():
        for b in tqdm(range(0, len(images), 256), desc="classifier features", mininterval=30):
            x = normalise(resize(to_tensor(images[b : b + 256]), hw))
            out.append(F.normalize(net.embed(x), dim=-1).numpy().astype(np.float32))
    emb = np.concatenate(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)
    return emb
