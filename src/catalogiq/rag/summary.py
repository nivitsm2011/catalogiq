"""Aspect-level product summaries. Every number is computed here; the LLM only phrases the facts.

For each product and aspect: how many reviews mention it, the share of those reviews that are
positive / mixed / negative (by star rating), and for fit the share that say it runs small, large or
true to size. A deterministic template sentence is always available; the LLM version is accepted
only if every number it writes appears in the computed facts.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml

from catalogiq.config import get_qa_config, resolve_path
from catalogiq.llm.client import LLMClient
from catalogiq.rag.aspects import aspect_names, rule_tag

LABELS = {
    "fit_size": "Fit and size",
    "fabric_material": "Fabric and material",
    "colour_accuracy": "Colour",
    "quality_durability": "Quality and durability",
    "comfort": "Comfort",
    "delivery_packaging": "Delivery and packaging",
}


def tag_review_aspects(chunks: pd.DataFrame, tags: np.ndarray | None = None) -> pd.DataFrame:
    """Review-level aspect table: a review mentions an aspect if any of its chunks does.

    Args:
        chunks: Chunk table (one row per chunk).
        tags: Optional precomputed boolean (n_chunks, n_aspects) matrix; keyword rules by default.
    """
    names = aspect_names()
    matrix = rule_tag(chunks["text"].tolist()) if tags is None else tags
    frame = pd.DataFrame(matrix, columns=names)
    frame["review_id"] = chunks["review_id"].to_numpy()
    by_review = frame.groupby("review_id")[names].any()
    meta = chunks.drop_duplicates("review_id").set_index("review_id")[
        ["item_id", "rating", "verified_purchase", "helpful_vote"]
    ]
    return meta.join(by_review).reset_index()


@lru_cache(maxsize=1)
def _fit_patterns() -> dict[str, re.Pattern[str]]:
    cfg = get_qa_config()["fit_direction"]
    return {k: re.compile("|".join(v), re.IGNORECASE) for k, v in cfg.items()}


def fit_direction(texts: list[str]) -> dict[str, float | int]:
    """Counts and shares of reviews saying runs small / runs large / true to size.

    A review that matches both a size complaint and "true to size" counts as the complaint, since
    the complaint is the more specific statement. Reviews with no direction are not counted.
    """
    pats = _fit_patterns()
    counts = {"runs_small": 0, "runs_large": 0, "true_to_size": 0}
    for t in texts:
        small, large, tts = (
            bool(pats[k].search(t)) for k in ("runs_small", "runs_large", "true_to_size")
        )
        if small and not large:
            counts["runs_small"] += 1
        elif large and not small:
            counts["runs_large"] += 1
        elif tts and not small and not large:
            counts["true_to_size"] += 1
    total = sum(counts.values())
    out: dict[str, float | int] = {**counts, "with_direction": total}
    for k, v in counts.items():
        out[f"pct_{k}"] = round(100 * v / total) if total else 0
    return out


def product_facts(
    item_id: str, review_aspects: pd.DataFrame, review_text: dict[str, str]
) -> dict[str, Any]:
    """Computed facts for one product: per-aspect mention counts and rating-based sentiment."""
    cfg = get_qa_config()["summary"]
    rev = review_aspects[review_aspects["item_id"] == item_id]
    facts: dict[str, Any] = {"item_id": item_id, "n_reviews": int(len(rev)), "aspects": {}}
    for a in aspect_names():
        m = rev[rev[a]]
        if len(m) < cfg["min_mentions"]:
            continue
        pos = int((m["rating"] >= cfg["positive_min_rating"]).sum())
        neg = int((m["rating"] <= cfg["negative_max_rating"]).sum())
        entry: dict[str, Any] = {
            "mentions": int(len(m)),
            "pct_of_reviews": round(100 * len(m) / max(len(rev), 1)),
            "pct_positive": round(100 * pos / len(m)),
            "pct_negative": round(100 * neg / len(m)),
            "pct_mixed": round(100 * (len(m) - pos - neg) / len(m)),
            "mean_rating": round(float(m["rating"].mean()), 1),
        }
        if a == "fit_size":
            entry["fit"] = fit_direction([review_text[r] for r in m["review_id"]])
        facts["aspects"][a] = entry
    return facts


def aspect_sentence(aspect: str, e: dict[str, Any]) -> str:
    """Deterministic sentence for one aspect, built straight from its numbers."""
    s = (
        f"{LABELS[aspect]}: {e['mentions']} reviews mention it; {e['pct_positive']}% are positive "
        f"and {e['pct_negative']}% negative."
    )
    fit = e.get("fit")
    if fit and fit["with_direction"] >= 3:
        s += (
            f" Of {fit['with_direction']} reviews that say how it fits, "
            f"{fit['pct_runs_small']}% say it runs small, {fit['pct_runs_large']}% large and "
            f"{fit['pct_true_to_size']}% true to size."
        )
    return s


def template_summary(facts: dict[str, Any]) -> str:
    """Deterministic summary: one sentence per aspect."""
    lines = [aspect_sentence(a, e) for a, e in facts["aspects"].items()]
    return " ".join(lines) if lines else "Not enough reviews mention any aspect to summarise."


def aspect_facts_text(aspect: str, e: dict[str, Any]) -> str:
    """Plain-language fact line for one aspect (what the LLM is shown)."""
    line = (
        f"{e['mentions']} reviews mention it ({e['pct_of_reviews']}% of all reviews). "
        f"Of those: {e['pct_positive']}% positive, {e['pct_mixed']}% mixed, "
        f"{e['pct_negative']}% negative; average {e['mean_rating']} stars."
    )
    fit = e.get("fit")
    if fit and fit["with_direction"] >= 3:
        line += (
            f" {fit['with_direction']} reviews say how it fits: "
            f"{fit['pct_runs_small']}% runs small, "
            f"{fit['pct_runs_large']}% runs large, {fit['pct_true_to_size']}% true to size."
        )
    return line


def facts_text(facts: dict[str, Any]) -> str:
    """All aspect fact lines for the prompt."""
    return "\n".join(f"- {a}: {aspect_facts_text(a, e)}" for a, e in facts["aspects"].items())


_LEADING_LABEL = re.compile(r"^[A-Za-z][A-Za-z /&-]{2,40}:\s*")


def strip_label(text: str) -> str:
    """Drop a label the model repeated at the start ("Comfort: ..."); the code adds its own."""
    return _LEADING_LABEL.sub("", text.strip(), count=1)


def numbers_in(text: str) -> set[str]:
    """All integers/decimals appearing in ``text`` (as strings)."""
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def phrase_summary(facts: dict[str, Any], title: str, llm: LLMClient) -> dict[str, Any]:
    """LLM-phrased summary, checked sentence by sentence.

    The model writes one sentence per aspect. A sentence may only contain numbers that appear in
    THAT aspect's facts; otherwise that aspect falls back to the deterministic sentence.

    Returns:
        ``summary`` text, ``source`` (llm / template / mixed), ``bad_aspects`` and ``bad_numbers``.
    """
    aspects = facts["aspects"]
    if not aspects:
        return {
            "summary": template_summary(facts),
            "source": "template",
            "bad_aspects": [],
            "bad_numbers": [],
        }
    path = resolve_path(get_qa_config()["paths"]["prompts_summary"])
    p = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = list(aspects)
    msgs = [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user"].format(title=title, facts=facts_text(facts))},
    ]
    schema = {
        "type": "object",
        "properties": {a: {"type": "string"} for a in names},
        "required": names,
    }
    try:
        data = json.loads(llm.chat(msgs, schema=schema, temperature=0.0).text)
    except Exception:  # noqa: BLE001 - any failure falls back to the deterministic text
        return {
            "summary": template_summary(facts),
            "source": "template",
            "bad_aspects": names,
            "bad_numbers": [],
        }
    sentences, bad_aspects, bad_numbers = [], [], []
    for a in names:
        text = strip_label(str(data.get(a, "")))
        allowed = numbers_in(aspect_facts_text(a, aspects[a]))
        bad = sorted(numbers_in(text) - allowed)
        if not text or bad:
            sentences.append(aspect_sentence(a, aspects[a]))
            bad_aspects.append(a)
            bad_numbers += bad
        else:
            sentences.append(f"{LABELS[a]}: {text}")
    source = "template" if len(bad_aspects) == len(names) else "mixed" if bad_aspects else "llm"
    return {
        "summary": " ".join(sentences),
        "source": source,
        "bad_aspects": bad_aspects,
        "bad_numbers": bad_numbers,
    }
