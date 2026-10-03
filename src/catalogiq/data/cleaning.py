"""Business-driven cleaning of the fashion catalog (see docs/decisions.md for the reasoning)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

OTHER = "Other"


def load_colour_map(path: str | Path) -> dict[str, str]:
    """Load the palette YAML and invert it to ``raw colour -> colour family``."""
    with Path(path).open("r", encoding="utf-8") as handle:
        families: dict[str, list[str]] = yaml.safe_load(handle)
    return {raw: family for family, raws in families.items() for raw in raws}


def normalise_colours(series: pd.Series, mapping: dict[str, str]) -> pd.Series:
    """Map raw colour names to the retail palette; unmapped colours become ``Other``.

    Missing values stay missing (we do not guess a colour).
    """
    unmapped = sorted(set(series.dropna().unique()) - set(mapping))
    if unmapped:
        logger.info("colours not in palette (-> %s): %s", OTHER, unmapped)
    mapped = series.map(mapping)
    return mapped.where(~(series.notna() & mapped.isna()), OTHER).where(series.notna())


def handle_rare_classes(
    df: pd.DataFrame, target: str, min_count: int, policy: str
) -> tuple[pd.DataFrame, dict]:
    """Drop or merge classes that are too small to learn or to promise to sellers.

    Args:
        df: Table containing ``target``.
        target: Column to process.
        min_count: Classes with fewer rows are "rare".
        policy: ``"drop"`` removes rare rows; ``"other"`` relabels them as ``Other``.

    Returns:
        The processed frame and a log dict (classes affected, rows affected).
    """
    counts = df[target].value_counts()
    rare = counts[counts < min_count].index.tolist()
    affected = int(df[target].isin(rare).sum())
    if policy == "drop":
        df = df[~df[target].isin(rare)].copy()
    elif policy == "other":
        df = df.copy()
        df.loc[df[target].isin(rare), target] = OTHER
    else:
        raise ValueError(f"unknown rare-class policy: {policy}")
    info = {
        "target": target,
        "policy": policy,
        "min_count": min_count,
        "rare_classes": len(rare),
        "rows_affected": affected,
    }
    logger.info("rare classes | %s", info)
    return df, info


def clean_fashion(
    audited: pd.DataFrame,
    colour_map: dict[str, str],
    rare_cfg: dict[str, dict],
    taxonomy: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the full cleaning pipeline to the audited table.

    Steps (each logged with row counts): drop duplicate ids; drop missing/unreadable images;
    drop rows missing masterCategory/subCategory/articleType (the core taxonomy);
    normalise colours; merge or drop rare classes per target.

    Args:
        audited: Output of ``inspect_images`` (+ ``dup_group``).
        colour_map: Raw colour to palette family mapping.
        rare_cfg: ``fashion_cleaning.rare`` config.
        taxonomy: Target column names.

    Returns:
        Cleaned frame and a step-by-step log (list of dicts).
    """
    log: list[dict] = []
    df = audited.copy()

    def step(name: str, before: int, after: int) -> None:
        log.append({"step": name, "rows_before": before, "rows_after": after})
        logger.info("clean | %-38s %6d -> %6d", name, before, after)

    n = len(df)
    df = df.drop_duplicates(subset="id", keep="first")
    step("drop duplicate ids", n, len(df))

    n = len(df)
    df = df[df["image_exists"] & df["image_ok"]]
    step("drop missing/unreadable images", n, len(df))

    n = len(df)
    core = ["masterCategory", "subCategory", "articleType"]
    df = df.dropna(subset=core)
    step("drop rows missing core taxonomy", n, len(df))

    df = df.copy()
    df["baseColour_raw"] = df["baseColour"]
    df["baseColour"] = normalise_colours(df["baseColour"], colour_map)

    # Drop-policy targets first, repeated until stable: removing rare rows for one target can push
    # a class of another target below its threshold. Merge-policy targets run last so their
    # counts reflect the final set of rows.
    drop_targets = [t for t in taxonomy if rare_cfg[t]["policy"] == "drop"]
    other_targets = [t for t in taxonomy if rare_cfg[t]["policy"] != "drop"]
    for round_no in range(1, 11):
        n_round = len(df)
        for target in drop_targets:
            n = len(df)
            df, info = handle_rare_classes(df, target, rare_cfg[target]["min_count"], "drop")
            step(f"rare classes: {target} (drop, round {round_no})", n, len(df))
            log[-1].update({k: info[k] for k in ("rare_classes", "rows_affected")})
        if len(df) == n_round:
            break
    for target in other_targets:
        n = len(df)
        df, info = handle_rare_classes(df, target, rare_cfg[target]["min_count"], "other")
        step(f"rare classes: {target} (other)", n, len(df))
        log[-1].update({k: info[k] for k in ("rare_classes", "rows_affected")})
    return df.reset_index(drop=True), log
