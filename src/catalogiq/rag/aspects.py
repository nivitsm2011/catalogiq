"""Aspect tagging for review chunks: keyword rules (explainable) and three zero-shot alternatives.

Aspects: fit/size, fabric/material, colour accuracy, quality/durability, comfort,
delivery/packaging.
A chunk may carry several aspects ("mentions this aspect", independent of whether it is praise or
complaint). ``compare_taggers`` scores each method against the hand-labelled sample.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


def aspect_names() -> list[str]:
    """Aspect names in the fixed column order used everywhere."""
    return list(get_qa_config()["aspects"]["names"])


@lru_cache(maxsize=1)
def _rules() -> dict[str, re.Pattern[str]]:
    rules = get_qa_config()["aspects"]["rules"]
    return {a: re.compile("|".join(rules[a]), re.IGNORECASE) for a in aspect_names()}


def rule_matches(text: str) -> dict[str, list[str]]:
    """Which keyword hits each aspect has in ``text`` (empty list = aspect absent)."""
    return {a: [m.group(0) for m in p.finditer(text)] for a, p in _rules().items()}


def rule_tag(texts: Sequence[str]) -> np.ndarray:
    """Boolean matrix (n, n_aspects): True where a keyword rule matches."""
    names = aspect_names()
    rules = _rules()
    return np.array([[bool(rules[a].search(t)) for a in names] for t in texts], dtype=bool)


def embedding_scores(texts: Sequence[str]) -> np.ndarray:
    """Cosine similarity of each text to each aspect description (MiniLM), shape (n, n_aspects)."""
    from catalogiq.rag.knowledge_base import embed_texts

    desc = get_qa_config()["zero_shot"]["embed_descriptions"]
    a_emb = embed_texts([desc[a] for a in aspect_names()])
    return embed_texts(list(texts)) @ a_emb.T


@lru_cache(maxsize=1)
def _nli_pipeline() -> Any:
    from transformers import pipeline

    from catalogiq.utils import configure_cache_dirs

    configure_cache_dirs()
    name = get_qa_config()["zero_shot"]["nli_model"]
    logger.info("Loading NLI model %s", name)
    return pipeline("zero-shot-classification", model=name, device=-1)


def nli_scores(texts: Sequence[str]) -> np.ndarray:
    """Entailment probability per aspect (multi-label zero-shot NLI), shape (n, n_aspects)."""
    zs = get_qa_config()["zero_shot"]
    labels = [zs["nli_labels"][a] for a in aspect_names()]
    pipe = _nli_pipeline()
    out = pipe(
        list(texts),
        candidate_labels=labels,
        hypothesis_template=zs["hypothesis_template"],
        multi_label=True,
        batch_size=zs["batch_size"],
    )
    scores = np.zeros((len(texts), len(labels)))
    for i, res in enumerate(out if isinstance(out, list) else [out]):
        by_label = dict(zip(res["labels"], res["scores"], strict=True))
        scores[i] = [by_label[lab] for lab in labels]
    return scores


@lru_cache(maxsize=1)
def _llm_prompt() -> dict[str, str]:
    path = resolve_path(get_qa_config()["paths"]["prompts_zeroshot"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def llm_tag(texts: Sequence[str], llm: Any) -> np.ndarray:
    """Zero-shot tagging with the local LLM (one JSON call per text); boolean (n, n_aspects)."""
    zs = get_qa_config()["zero_shot"]
    names = aspect_names()
    p = _llm_prompt()
    schema = {
        "type": "object",
        "properties": {a: {"type": "boolean"} for a in names},
        "required": names,
    }
    out = np.zeros((len(texts), len(names)), dtype=bool)
    for i, t in enumerate(texts):
        msgs = [
            {"role": "system", "content": p["system"]},
            {"role": "user", "content": p["user"].format(review=t[: zs["llm_max_chars"]])},
        ]
        try:
            data = json.loads(llm.chat(msgs, schema=schema, temperature=0.0).text)
            out[i] = [bool(data.get(a, False)) for a in names]
        except Exception as exc:  # noqa: BLE001 - one bad reply must not stop the comparison
            logger.warning("LLM tagging failed for item %d: %s", i, exc)
    return out


def prf(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-aspect precision, recall, F1 (plus micro and macro rows)."""
    names = aspect_names()
    rows = []
    for j, a in enumerate(names):
        tp = int((y_true[:, j] & y_pred[:, j]).sum())
        fp = int((~y_true[:, j] & y_pred[:, j]).sum())
        fn = int((y_true[:, j] & ~y_pred[:, j]).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        rows.append(
            {"aspect": a, "support": int(y_true[:, j].sum()), "precision": p, "recall": r, "f1": f}
        )
    tp, fp, fn = (
        int((y_true & y_pred).sum()),
        int((~y_true & y_pred).sum()),
        int((y_true & ~y_pred).sum()),
    )
    mp, mr = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
    rows.append(
        {
            "aspect": "MICRO",
            "support": int(y_true.sum()),
            "precision": mp,
            "recall": mr,
            "f1": 2 * mp * mr / (mp + mr) if mp + mr else 0.0,
        }
    )
    macro = pd.DataFrame(rows[:-1])[["precision", "recall", "f1"]].mean()
    rows.append({"aspect": "MACRO", "support": int(y_true.sum()), **macro.to_dict()})
    return pd.DataFrame(rows)


def best_threshold(scores: np.ndarray, y_true: np.ndarray, grid: Sequence[float]) -> float:
    """Single threshold (shared by all aspects) maximising micro-F1 on the given labelled rows."""
    best, best_f = grid[0], -1.0
    for t in grid:
        f = float(prf(y_true, scores >= t).query("aspect == 'MICRO'")["f1"].iloc[0])
        if f > best_f:
            best, best_f = float(t), f
    return best


def tag_with(
    method: str, texts: Sequence[str], threshold: float | None = None, llm: Any = None
) -> np.ndarray:
    """Tag ``texts`` with one of ``rules``, ``embedding``, ``nli`` or ``llm`` (boolean matrix)."""
    zs = get_qa_config()["zero_shot"]
    if method == "rules":
        return rule_tag(texts)
    if method == "embedding":
        return embedding_scores(texts) >= (threshold or zs["embed_threshold"])
    if method == "nli":
        return nli_scores(texts) >= (threshold or zs["threshold"])
    if method == "llm":
        return llm_tag(texts, llm)
    raise ValueError(f"unknown aspect method {method!r}")


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Run ``fn`` and return (result, seconds)."""
    t0 = time.perf_counter()
    res = fn()
    return res, time.perf_counter() - t0
