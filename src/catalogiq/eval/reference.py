"""Drift check: compare a re-run with the numbers earlier phases recorded."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yaml

from catalogiq.config import get_eval_config, resolve_path


def load_reference() -> dict[str, dict[str, Any]]:
    """``configs/eval_reference.yaml`` -> ``{metric: {value, tol, source}}``."""
    path = resolve_path(get_eval_config()["paths"]["reference"])
    return yaml.safe_load(path.read_text(encoding="utf-8"))["reference"]


def tolerance(spec: dict[str, Any], value: float) -> tuple[float, str]:
    """Allowed absolute difference and the rule that gives it."""
    cfg = get_eval_config()["drift"]
    tol = spec["tol"]
    if isinstance(tol, (int, float)):
        return float(tol), "abs"
    if tol == "timing":
        return cfg["tolerance_rel_timing"] * abs(spec["value"]), "timing"
    if tol == "abs2":
        return cfg["tolerance_abs_2dp"], "abs"
    return cfg["tolerance_abs"], "abs"


def drift_table(
    metrics: dict[str, float], reference: dict[str, dict[str, Any]] | None = None
) -> pd.DataFrame:
    """One row per reference metric: previous value, new value, difference and status.

    Status is ``match`` (within the recording precision), ``timing`` (a timing inside its wide
    band), ``DRIFT`` (outside the tolerance) or ``not re-run`` (metric absent from this run).
    """
    reference = reference or load_reference()
    rows = []
    for key, spec in reference.items():
        prev = float(spec["value"])
        if key not in metrics or pd.isna(metrics[key]):
            rows.append(
                {
                    "metric": key,
                    "previous": prev,
                    "now": float("nan"),
                    "diff": float("nan"),
                    "status": "not re-run",
                    "source": spec["source"],
                }
            )
            continue
        now = float(metrics[key])
        tol, kind = tolerance(spec, now)
        diff = now - prev
        if abs(diff) <= tol:
            status = "timing" if kind == "timing" and abs(diff) > 1e-9 else "match"
        else:
            status = "DRIFT"
        rows.append(
            {
                "metric": key,
                "previous": prev,
                "now": now,
                "diff": diff,
                "status": status,
                "source": spec["source"],
            }
        )
    return pd.DataFrame(rows)
