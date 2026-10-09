"""Evaluate image search, text search and latency.

Protocol: the gallery is the train+val catalog (35.9k photos) and the queries are test photos, so no
near-duplicate twin of a query can be in the gallery (Phase 1 split is group-aware). A result is
relevant if it has the same articleType AND the same baseColour as the query.

Usage: python scripts/evaluate_search.py
Outputs: reports/search_metrics.csv, reports/text_search_metrics.csv, reports/search_latency.json,
         reports/clip_zero_shot_padded.csv
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from catalogiq.config import get_search_config, get_vision_config, resolve_path
from catalogiq.logging_utils import get_logger
from catalogiq.search.embeddings import ClipEncoder, classifier_embeddings, embed_catalog
from catalogiq.search.engine import SearchEngine
from catalogiq.search.index import l2_normalise
from catalogiq.search.metrics import (
    average_precision_at_k,
    count_relevant,
    precision_at_k,
    recall_at_k,
    relevance_matrix,
)
from catalogiq.utils import set_seed

logger = get_logger(__name__)
KEYS = ["articleType", "baseColour"]


def evaluate_representation(
    name: str,
    emb: np.ndarray,
    catalog: pd.DataFrame,
    q_pos: np.ndarray,
    g_pos: np.ndarray,
    ks,
    map_k,
) -> dict:
    """Image->image retrieval metrics for one embedding space."""
    gallery_cat = catalog.iloc[g_pos].reset_index(drop=True)
    queries = catalog.iloc[q_pos].reset_index(drop=True)
    engine = SearchEngine(gallery_cat, emb[g_pos])
    ranked = engine.search_matrix(emb[q_pos], max(max(ks), map_k))
    n_rel = count_relevant(queries, gallery_cat, KEYS)
    ok = n_rel > 0
    rel = relevance_matrix(queries[ok], gallery_cat, ranked[ok], KEYS)
    out = {
        "representation": name,
        "queries": int(ok.sum()),
        "queries_without_relevant": int((~ok).sum()),
    }
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(rel, k), 4)
    out[f"precision@{map_k}"] = round(precision_at_k(rel, map_k), 4)
    out[f"mAP@{map_k}"] = round(average_precision_at_k(rel, n_rel[ok], map_k), 4)
    # looser definition for context: same articleType only
    rel_type = relevance_matrix(queries, gallery_cat, ranked, ["articleType"])
    out[f"precision@{map_k}_articleType_only"] = round(precision_at_k(rel_type, map_k), 4)
    return out


def main() -> None:
    """Run all search evaluations and write the report files."""
    set_seed(42)
    scfg, vcfg = get_search_config(), get_vision_config()
    ecfg = scfg["evaluation"]
    image_emb, text_emb, catalog = embed_catalog()
    clf_emb = classifier_embeddings()
    frozen = np.load(resolve_path(vcfg["baselines"]["embeddings_path"]))["emb"]
    assert len(frozen) == len(catalog) == len(clf_emb), "embedding rows are not aligned"

    split = catalog["split"].to_numpy()
    g_pos = np.where(np.isin(split, ["train", "val"]))[0]
    test_pos = np.where(split == "test")[0]
    rng = np.random.default_rng(42)
    n_q = ecfg["n_queries"] or len(test_pos)
    q_pos = np.sort(rng.choice(test_pos, min(n_q, len(test_pos)), replace=False))
    logger.info("gallery %d photos, %d queries", len(g_pos), len(q_pos))

    rows = [
        evaluate_representation(
            "CLIP ViT-B/32 image embeddings",
            image_emb,
            catalog,
            q_pos,
            g_pos,
            ecfg["k_values"],
            ecfg["map_k"],
        ),
        evaluate_representation(
            "Phase 2 fine-tuned classifier features",
            clf_emb,
            catalog,
            q_pos,
            g_pos,
            ecfg["k_values"],
            ecfg["map_k"],
        ),
        evaluate_representation(
            "Frozen ImageNet MobileNetV3 features",
            frozen,
            catalog,
            q_pos,
            g_pos,
            ecfg["k_values"],
            ecfg["map_k"],
        ),
        evaluate_representation(
            "Random vectors (floor)",
            rng.normal(size=image_emb.shape).astype(np.float32),
            catalog,
            q_pos,
            g_pos,
            ecfg["k_values"],
            ecfg["map_k"],
        ),
    ]
    metrics = pd.DataFrame(rows)
    metrics.to_csv(resolve_path("reports/search_metrics.csv"), index=False)
    logger.info("image search:\n%s", metrics.to_string(index=False))

    # ----------------------------------------------------------- text -> image
    encoder = ClipEncoder()
    gallery_cat = catalog.iloc[g_pos].reset_index(drop=True)
    engine = SearchEngine(gallery_cat, image_emb[g_pos], encoder=encoder)
    combo = (
        gallery_cat.groupby(["baseColour", "articleType", "gender"])
        .size()
        .rename("n")
        .reset_index()
    )
    combo = combo[(combo["n"] >= ecfg["text_min_relevant"]) & (combo["baseColour"] != "Other")]
    combo = combo.sample(min(ecfg["text_queries"], len(combo)), random_state=42).reset_index(
        drop=True
    )
    texts = [
        f"{r.baseColour.lower()} {r.articleType.lower()} for {r.gender.lower()}"
        for r in combo.itertuples()
    ]
    q_vec = encoder.encode_texts(texts)
    ranked_text = engine.search_matrix(q_vec, 10)
    text_keys = ["baseColour", "articleType", "gender"]
    rel_t = relevance_matrix(combo, gallery_cat, ranked_text, text_keys)
    n_rel_t = count_relevant(combo, gallery_cat, text_keys)
    rand_rank = rng.integers(0, len(gallery_cat), size=ranked_text.shape)
    rel_rand = relevance_matrix(combo, gallery_cat, rand_rank, text_keys)
    # text-to-text oracle: query text vs the attribute sentence embedded for every gallery item
    t_engine = SearchEngine(gallery_cat, text_emb[g_pos])
    rel_tt = relevance_matrix(combo, gallery_cat, t_engine.search_matrix(q_vec, 10), text_keys)
    text_rows = []
    for name, rel in [
        ("CLIP text -> image embeddings", rel_t),
        ("Random gallery items (floor)", rel_rand),
        ("CLIP text -> attribute-sentence embeddings (reference, not image-based)", rel_tt),
    ]:
        text_rows.append(
            {
                "method": name,
                "queries": len(combo),
                "recall@10": round(recall_at_k(rel, 10), 4),
                "precision@10": round(precision_at_k(rel, 10), 4),
                "mAP@10": round(average_precision_at_k(rel, n_rel_t, 10), 4),
            }
        )
    text_metrics = pd.DataFrame(text_rows)
    text_metrics.to_csv(resolve_path("reports/text_search_metrics.csv"), index=False)
    pd.DataFrame({"query": texts, "relevant_in_gallery": n_rel_t}).to_csv(
        resolve_path("reports/text_queries_used.csv"), index=False
    )
    logger.info("text search:\n%s", text_metrics.to_string(index=False))

    # ----------------------------------------------------------- zero-shot with padded images
    prompts = vcfg["baselines"]["clip_prompts"]
    labels = json.loads(
        (
            resolve_path(vcfg["inference"]["model_dir"]) / vcfg["inference"]["labels_file"]
        ).read_text()
    )
    test_cat = catalog.iloc[test_pos].reset_index(drop=True)
    zs_rows = []
    for target in ("articleType", "baseColour", "gender"):
        classes = labels[target]
        txt = encoder.encode_texts([prompts[target].format(c.lower()) for c in classes])
        pred = (l2_normalise(image_emb[test_pos]) @ txt.T).argmax(axis=1)
        truth = test_cat[target].map({c: i for i, c in enumerate(classes)})
        valid = truth.notna().to_numpy()
        from sklearn.metrics import f1_score

        zs_rows.append(
            {
                "target": target,
                "accuracy": round(float((pred[valid] == truth[valid].to_numpy()).mean()), 4),
                "macro_f1": round(
                    float(f1_score(truth[valid].astype(int), pred[valid], average="macro")), 4
                ),
            }
        )
    zs = pd.DataFrame(zs_rows)
    zs.to_csv(resolve_path("reports/clip_zero_shot_padded.csv"), index=False)
    logger.info("zero-shot with white-padded images:\n%s", zs.to_string(index=False))

    # ----------------------------------------------------------- latency
    runs = ecfg["latency_runs"]
    full = SearchEngine(catalog, image_emb, encoder=encoder)
    vecs = image_emb[rng.choice(len(image_emb), runs, replace=False)]
    t_search = []
    for v in vecs:
        t0 = time.perf_counter()
        full.search_vector(v, 10)
        t_search.append((time.perf_counter() - t0) * 1000)
    t_filtered = []
    for v in vecs[: runs // 2]:
        t0 = time.perf_counter()
        full.search_vector(v, 10, filters={"gender": "Men", "baseColour": "Black"})
        t_filtered.append((time.perf_counter() - t0) * 1000)
    sample_imgs = np.asarray(
        __import__("catalogiq.vision.data", fromlist=["x"]).build_image_cache()[0][:30]
    )
    encoder.encode_image(sample_imgs[0])
    t_img = []
    for a in sample_imgs[:20]:
        t0 = time.perf_counter()
        encoder.encode_image(a)
        t_img.append((time.perf_counter() - t0) * 1000)
    t_txt = []
    for q in texts[:20]:
        t0 = time.perf_counter()
        encoder.encode_text(q)
        t_txt.append((time.perf_counter() - t0) * 1000)

    def stats(x: list[float]) -> dict:
        return {
            "mean_ms": round(float(np.mean(x)), 2),
            "p95_ms": round(float(np.percentile(x, 95)), 2),
            "n": len(x),
        }

    latency = {
        "catalog_size": len(catalog),
        "vector_search_flat_unfiltered": stats(t_search),
        "vector_search_flat_filtered_gender_colour": stats(t_filtered),
        "clip_encode_one_image": stats(t_img),
        "clip_encode_one_text": stats(t_txt),
        "note": "CPU only, x64 Python emulated on ARM64; search_by_image = encode + vector search",
    }
    resolve_path("reports/search_latency.json").write_text(
        json.dumps(latency, indent=2), encoding="utf-8"
    )
    logger.info("latency: %s", json.dumps(latency))


if __name__ == "__main__":
    main()
