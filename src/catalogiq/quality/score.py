"""Transparent 0-100 listing quality score with a plain-English fix for every flag.

score = w_image * image + w_consistency * consistency + w_completeness * completeness  (each 0-100)
- image starts at 100 and loses a fixed penalty per raised image check (never below 0);
- consistency starts at 100 and loses a fixed penalty per mismatching attribute (never below 0);
- completeness is the share of required attributes present.
Weights and penalties live in ``configs/quality.yaml`` (``quality.score``); reasons are in
docs/decisions.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from catalogiq.config import get_quality_config


@dataclass
class Flag:
    """One issue found on a listing."""

    code: str
    area: str  # image | consistency | completeness
    penalty: float
    message: str  # what is wrong
    fix: str  # what the seller should do


def _fix(code: str, **kw: str) -> str:
    return get_quality_config()["fixes"][code].format(**kw)


def image_flags(raised: list[str]) -> list[Flag]:
    """Flags for the raised image checks."""
    cfg = get_quality_config()["score"]["image_penalty"]
    return [Flag(c, "image", cfg[c], c.replace("_", " "), _fix(c)) for c in raised]


def mismatch_flags(mismatches: list[dict[str, str]]) -> list[Flag]:
    """Flags for attribute mismatches (each dict: attribute, stated, seen)."""
    pen = get_quality_config()["score"]["consistency_penalty"]["attribute_mismatch"]
    return [
        Flag(
            "attribute_mismatch",
            "consistency",
            pen,
            f"{m['attribute']} listed as {m['stated']} but the photo looks like {m['seen']}",
            _fix(
                "attribute_mismatch",
                attribute=m["attribute"],
                detail=f"listed {m['stated']}, photo looks like {m['seen']}",
            ),
        )
        for m in mismatches
    ]


def completeness_flags(missing: list[str]) -> list[Flag]:
    """One flag when required attributes are missing."""
    if not missing:
        return []
    return [
        Flag(
            "missing_attributes",
            "completeness",
            0.0,
            "missing: " + ", ".join(missing),
            _fix("missing_attributes", detail=", ".join(missing)),
        )
    ]


def listing_score(
    raised_image_checks: list[str],
    mismatches: list[dict[str, str]],
    completeness_pct: float,
    missing: list[str],
) -> dict[str, Any]:
    """Combine the three components into the score and the list of flags with fixes."""
    cfg = get_quality_config()["score"]
    f_img, f_con, f_com = (
        image_flags(raised_image_checks),
        mismatch_flags(mismatches),
        completeness_flags(missing),
    )
    comp = {
        "image": max(0.0, 100.0 - sum(f.penalty for f in f_img)),
        "consistency": max(0.0, 100.0 - sum(f.penalty for f in f_con)),
        "completeness": float(completeness_pct),
    }
    w = cfg["weights"]
    score = sum(w[k] * comp[k] for k in comp) / sum(w.values())
    flags = f_img + f_con + f_com
    return {
        "score": round(score, 1),
        "components": comp,
        "flags": [asdict(f) for f in flags],
        "needs_attention": score < cfg["issue_below"],
    }
