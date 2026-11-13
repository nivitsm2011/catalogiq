"""Everything the API loads once at startup: models, indexes, tables."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import yaml

from catalogiq.config import (
    get_config,
    get_qa_config,
    get_quality_config,
    resolve_path,
)
from catalogiq.logging_utils import get_logger

logger = get_logger("api.state")
DEMO_N = 10


@dataclass
class AppState:
    """Loaded models and data. Tests build one from fakes; the server builds it with ``load``."""

    predictor: Any
    encoder: Any
    catalog: pd.DataFrame
    image_emb: np.ndarray
    engine: Any
    looks: Any
    generator: Any
    qa: Any
    chunks: pd.DataFrame
    review_items: pd.DataFrame
    llm: Any
    labels: dict[str, list[str]]
    text_prompts: np.ndarray
    value_emb: dict[str, np.ndarray]
    demo: list[dict[str, Any]]
    questions: list[dict[str, Any]]
    started: float = field(default_factory=time.time)
    llm_lock: threading.Lock = field(default_factory=threading.Lock)
    models: dict[str, bool] = field(default_factory=dict)
    version: str = "0.8.0"
    _lazy: dict[str, Any] = field(default_factory=dict)
    _lazy_lock: threading.Lock = field(default_factory=threading.Lock)

    def lazy(self, name: str, build: Any) -> Any:
        """Build a heavy object on first use (thread-safe) and keep it."""
        with self._lazy_lock:
            if name not in self._lazy:
                start = time.perf_counter()
                self._lazy[name] = build()
                logger.info("built %s in %.1fs", name, time.perf_counter() - start)
            return self._lazy[name]

    @property
    def review_text(self) -> pd.DataFrame:
        """All reviews with ``review_id`` and joined text, indexed by id (loaded on demand)."""
        from catalogiq.rag.eval_qa import review_texts

        return self.lazy("review_text", lambda: review_texts().set_index("review_id", drop=False))

    @classmethod
    def load(cls) -> AppState:
        """Load every model and table (about 30-60 seconds on this CPU)."""
        from catalogiq.llm.client import get_llm_client
        from catalogiq.quality.clip_checks import prompt_embeddings
        from catalogiq.quality.consistency import value_embeddings
        from catalogiq.rag.evaluate import (
            load_brand_lexicon,
            sample_products,
            seller_notes_for,
        )
        from catalogiq.rag.knowledge_base import get_client
        from catalogiq.rag.listing import ListingGenerator
        from catalogiq.rag.qa import ReviewQA
        from catalogiq.rag.retrieval import HybridRetriever
        from catalogiq.rag.reviews_index import load_chunks
        from catalogiq.search.embeddings import ClipEncoder, embed_catalog
        from catalogiq.search.engine import SearchEngine
        from catalogiq.search.looks import ShopTheLook, attributes_from_predictions
        from catalogiq.vision.predict import AttributePredictor

        t0 = time.perf_counter()
        cfg = get_config()
        predictor = AttributePredictor()
        encoder = ClipEncoder()
        image_emb, _, catalog = embed_catalog()
        engine = SearchEngine(catalog, image_emb, encoder=encoder)
        looks = ShopTheLook(engine, lambda img: attributes_from_predictions(predictor.predict(img)))
        llm = get_llm_client()
        generator = ListingGenerator(llm, HybridRetriever(), load_brand_lexicon())
        chunks = load_chunks()
        meta = pd.read_parquet(resolve_path(cfg["data"]["reviews"]["meta_parquet"]))
        titles = dict(zip(meta["parent_asin"], meta["title"], strict=True))
        gate_path = resolve_path(get_qa_config()["paths"]["eval_dir"]) / "gate.json"
        gate = (
            json.loads(gate_path.read_text(encoding="utf-8"))["min_similarity"]
            if gate_path.exists()
            else None
        )
        qa = ReviewQA(
            llm,
            chunks,
            get_client().get_collection(get_qa_config()["paths"]["collection"]),
            titles,
            gate,
        )
        labels = {t: predictor.labels[t] for t in predictor.targets}
        qdata = yaml.safe_load(
            resolve_path(get_qa_config()["paths"]["questions"]).read_text(encoding="utf-8")
        )
        products = sample_products().head(DEMO_N)
        attrs = json.loads(
            (resolve_path("reports/phase4") / "attributes.json").read_text(encoding="utf-8")
        )
        demo = []
        for p in products.to_dict("records"):
            pid = int(p["id"])
            demo.append(
                {
                    "id": pid,
                    "title": p["productDisplayName"],
                    "group": p["group"],
                    "attributes": {
                        k: {"label": v[0], "confidence": float(v[1]), "needs_review": bool(v[2])}
                        for k, v in attrs[str(pid)].items()
                    },
                    "seller_notes": {
                        str(k): str(v) for k, v in seller_notes_for(pd.Series(p)).items()
                    },
                }
            )
        state = cls(
            predictor=predictor,
            encoder=encoder,
            catalog=catalog,
            image_emb=image_emb,
            engine=engine,
            looks=looks,
            generator=generator,
            qa=qa,
            chunks=chunks,
            review_items=pd.read_parquet(resolve_path(cfg["data"]["reviews"]["meta_parquet"])),
            llm=llm,
            labels=labels,
            text_prompts=prompt_embeddings(encoder),
            value_emb=value_embeddings(encoder, labels),
            demo=demo,
            questions=qdata["questions"],
        )
        state.models = {
            "attribute_classifier": True,
            "clip_encoder": True,
            "search_index": True,
            "listing_generator": True,
            "review_qa": True,
        }
        logger.info(
            "state loaded in %.1fs (%d catalog items)", time.perf_counter() - t0, len(catalog)
        )
        return state

    # ---- settings the services read
    @staticmethod
    def quality_paths() -> dict[str, Any]:
        """Paths of the saved quality report (for the ops dashboard)."""
        return get_quality_config()["paths"]
