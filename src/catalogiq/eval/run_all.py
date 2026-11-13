"""Unified evaluation: re-run the Phase 2-6 evaluations, log to MLflow, write the scorecard.

Usage (from the project root):
    python -m catalogiq.eval.run_all                     # everything (about 1 hour on this CPU)
    python -m catalogiq.eval.run_all --only vision,qa    # some components (merged into results)
    python -m catalogiq.eval.run_all --reuse-image-eval  # skip the 20-minute degraded-photo run
    python -m catalogiq.eval.run_all --skip-system       # skip the uncached latency/RAM run
    python -m catalogiq.eval.run_all --render-only       # rebuild the scorecard from saved results
Everything is logged to MLflow experiment ``catalogiq-eval`` (one parent run tagged with the git
commit hash, one nested run per component). Put the owner's timings in
reports/phase7/manual_listing_times.csv and use --render-only to refresh the impact section.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from typing import Any

import mlflow
import pandas as pd

from catalogiq.config import get_config, get_eval_config, resolve_path
from catalogiq.eval import common
from catalogiq.eval.common import ComponentResult, git_info, results_dir
from catalogiq.eval.components import RUNNERS
from catalogiq.eval.impact import build_impact, load_timings
from catalogiq.eval.reference import drift_table
from catalogiq.eval.scorecard import render
from catalogiq.eval.system import disk_footprint, end_to_end
from catalogiq.logging_utils import get_logger
from catalogiq.utils import set_seed

logger = get_logger("run_all")
ORDER = ["vision", "search", "duplicates", "listing", "qa", "quality"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return (
            value.reset_index().to_dict("records") if value.index.name else value.to_dict("records")
        )
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def result_to_dict(res: ComponentResult) -> dict[str, Any]:
    """A component result in a JSON-friendly form (tables become lists of records)."""
    return {
        "metrics": {k: (None if math.isnan(v) else v) for k, v in res.metrics.items()},
        "tables": _jsonable(res.tables),
        "provenance": _jsonable(res.provenance),
        "seconds": res.seconds,
    }


def _mlflow_name(key: str) -> str:
    return key.replace("@", "_at_")


def _finite(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        _mlflow_name(k): float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and math.isfinite(v)
    }


def _flat_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for c in results["components"].values():
        flat.update({k: v for k, v in c["metrics"].items() if v is not None})
    return flat


def _finish(results: dict[str, Any]) -> str:
    """Drift table, impact section, scorecard file, results file; returns the scorecard text."""
    cfg = get_eval_config()
    metrics = _flat_metrics(results)
    drift = drift_table(metrics)
    results["drift"] = drift.to_dict("records")
    results["impact"] = build_impact(metrics, results.get("system"), load_timings())
    results["generated"] = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    notes = cfg.get("drift_notes", [])
    text = render(results, notes)
    resolve_path(cfg["paths"]["scorecard"]).write_text(text, encoding="utf-8")
    resolve_path(cfg["paths"]["results"]).write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8"
    )
    drift.to_csv(resolve_path(cfg["paths"]["drift"]), index=False)
    return text


def run_all(
    only: list[str] | None = None,
    reuse_image_eval: bool = False,
    skip_system: bool = False,
    render_only: bool = False,
    allow_llm_calls: int | None = 25,
    resume: bool = False,
) -> dict[str, Any]:
    """Run the selected components, log to MLflow and write the scorecard."""
    cfg = get_eval_config()
    results_path = resolve_path(cfg["paths"]["results"])
    results_dir()
    results: dict[str, Any] = (
        json.loads(results_path.read_text(encoding="utf-8"))
        if results_path.exists()
        else {"components": {}}
    )
    git = git_info()
    results["git"] = git
    if render_only:
        _finish(results)
        logger.info("scorecard rebuilt from saved results")
        return results
    set_seed(cfg["seed"])
    common.LLM_CALL_BUDGET["max_calls"] = allow_llm_calls
    names = [n for n in ORDER if only is None or n in only]
    if resume:
        done = set(results["components"])
        names = [n for n in names if n not in done]
        logger.info("resuming: skipping %s", sorted(done) or "nothing")
    # MLflow >= 3.16 needs an explicit opt-in for the local file store the brief asks for
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    store = resolve_path(get_config()["paths"]["mlflow_tracking_dir"])
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.as_uri())
    mlflow.set_experiment(cfg["experiment"])
    start = time.perf_counter()
    with mlflow.start_run(run_name=f"eval-{git['short']}"):
        mlflow.set_tags(
            {
                "git_commit": git["commit"],
                "git_commit_short": git["short"],
                "git_branch": git["branch"],
                "git_dirty": str(git["dirty"]),
                "components": ",".join(names),
            }
        )
        for name in names:
            logger.info("=== %s", name)
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.set_tags({"git_commit": git["commit"], "component": name})
                kwargs = {"reuse_image_eval": reuse_image_eval} if name == "quality" else {}
                res = RUNNERS[name](**kwargs)
                mlflow.log_metrics(_finite(res.metrics))
                mlflow.log_metric("eval_seconds", res.seconds)
                results["components"][name] = result_to_dict(res)
                results.setdefault("run_commit", git["commit"])
                results_path.write_text(
                    json.dumps(results, indent=1, default=str), encoding="utf-8"
                )
                logger.info("%s done in %.0f s, %d metrics", name, res.seconds, len(res.metrics))
        if not skip_system and (only is None or "system" in only):
            logger.info("=== system")
            with mlflow.start_run(run_name="system", nested=True):
                mlflow.set_tags({"git_commit": git["commit"], "component": "system"})
                sysres = end_to_end()
                results["system"] = _jsonable(sysres)
                ram = sysres["ram"]
                mlflow.log_metrics(
                    {
                        "ram_peak_python_mb": ram["python_mb"],
                        "ram_peak_ollama_mb": ram["ollama_mb"],
                        "ram_peak_combined_mb": ram["combined_mb"],
                    }
                )
                for mode, s in sysres["summary"].items():
                    mlflow.log_metrics(
                        {
                            f"{mode}_total_mean_s": s["total_s"]["mean"],
                            f"{mode}_total_p95_s": s["total_s"]["p95"],
                        }
                    )
                disk = disk_footprint()
                results["disk"] = _jsonable(disk)
                results_path.write_text(
                    json.dumps(results, indent=1, default=str), encoding="utf-8"
                )
                mlflow.log_metrics({"disk_project_gb": disk["project_total_gb"]})
        elif "disk" not in results:
            results["disk"] = _jsonable(disk_footprint())
        results["total_seconds"] = time.perf_counter() - start
        text = _finish(results)
        mlflow.log_artifact(str(resolve_path(cfg["paths"]["scorecard"])))
        mlflow.log_artifact(str(resolve_path(cfg["paths"]["results"])))
        drift = pd.DataFrame(results["drift"])
        mlflow.log_metrics(
            {
                "drift_matches": float((drift["status"] == "match").sum()),
                "drift_timing": float((drift["status"] == "timing").sum()),
                "drift_drifted": float((drift["status"] == "DRIFT").sum()),
                "drift_not_rerun": float((drift["status"] == "not re-run").sum()),
            }
        )
    logger.info("scorecard written to %s (%d chars)", cfg["paths"]["scorecard"], len(text))
    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only", default=None, help="comma-separated components: " + ",".join(ORDER + ["system"])
    )
    parser.add_argument("--reuse-image-eval", action="store_true")
    parser.add_argument("--skip-system", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="skip components already in the saved results"
    )
    parser.add_argument(
        "--allow-llm-calls",
        type=int,
        default=25,
        help="LLM cache misses allowed before stopping (default 25; each takes 10-100 s on CPU)",
    )
    args = parser.parse_args()
    run_all(
        only=args.only.split(",") if args.only else None,
        reuse_image_eval=args.reuse_image_eval,
        skip_system=args.skip_system,
        render_only=args.render_only,
        allow_llm_calls=args.allow_llm_calls,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
