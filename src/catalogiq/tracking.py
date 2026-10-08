"""MLflow setup (local file store inside the project) and small logging helpers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import mlflow

from catalogiq.config import get_config, resolve_path


@contextmanager
def start_run(run_name: str, params: dict[str, Any] | None = None) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run on the local file store at ``paths.mlflow_tracking_dir``.

    Args:
        run_name: Human-readable run name.
        params: Parameters to log immediately (values are stringified by MLflow).
    """
    # MLflow >= 3.16 refuses the plain-file backend unless explicitly allowed. The project brief
    # asks for a local file store, so we opt in (docs/decisions.md lists the SQLite alternative).
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    cfg = get_config()
    store = resolve_path(cfg["paths"]["mlflow_tracking_dir"])
    store.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(store.as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    with mlflow.start_run(run_name=run_name) as run:
        if params:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        yield run
