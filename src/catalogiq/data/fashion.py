"""Loading the Kaggle Fashion Product Images (Small) metadata.

``styles.csv`` contains rows where ``productDisplayName`` holds unquoted commas, so a plain
``pd.read_csv`` either errors or drops them. We split each line on the first ``n_cols - 1``
commas instead: the free-text name is always the *last* column, so surplus commas belong to it.
Every repaired and every unrecoverable row is counted and logged, never silently dropped.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)

TEXT_COLUMN = "productDisplayName"


@dataclass
class LoadReport:
    """Counts describing how ``styles.csv`` was parsed."""

    lines_total: int
    rows_clean: int
    rows_repaired: int
    rows_unrecoverable: int
    unrecoverable_line_numbers: list[int]

    def as_dict(self) -> dict:
        """Return the report as a plain dictionary (for JSON logging)."""
        return asdict(self)


def load_styles(path: str | Path) -> tuple[pd.DataFrame, LoadReport]:
    """Parse ``styles.csv`` without silently losing malformed lines.

    Args:
        path: Location of ``styles.csv``.

    Returns:
        The parsed DataFrame (all columns as strings, empty cells as NaN) and a ``LoadReport``.
    """
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as handle:
        lines = handle.read().splitlines()
    header = lines[0].lstrip("﻿").split(",")
    n_cols = len(header)
    rows: list[list[str]] = []
    repaired = 0
    bad_lines: list[int] = []
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split(",", maxsplit=n_cols - 1)
        if len(parts) < n_cols:
            bad_lines.append(line_no)
            continue
        if line.count(",") >= n_cols:
            repaired += 1
        rows.append(parts)
    df = pd.DataFrame(rows, columns=header).replace(r"^\s*$", pd.NA, regex=True)
    df["id"] = df["id"].astype(int)
    report = LoadReport(
        lines_total=len(lines) - 1,
        rows_clean=len(rows) - repaired,
        rows_repaired=repaired,
        rows_unrecoverable=len(bad_lines),
        unrecoverable_line_numbers=bad_lines,
    )
    logger.info("styles.csv parse report: %s", report.as_dict())
    if bad_lines:
        logger.warning("%d lines had too few columns and were NOT loaded", len(bad_lines))
    return df, report


def image_path(images_dir: str | Path, product_id: int) -> Path:
    """Return the expected image path for a product id."""
    return Path(images_dir) / f"{product_id}.jpg"
