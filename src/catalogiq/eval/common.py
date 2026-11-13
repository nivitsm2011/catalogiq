"""Shared helpers for the unified evaluation: git info, script loading, LLM call counting."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd

from catalogiq.config import PROJECT_ROOT, get_eval_config, resolve_path
from catalogiq.llm.client import LLMClient, LLMResponse
from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ComponentResult:
    """Everything one component evaluation produces.

    Attributes:
        name: Component key (vision, search, duplicates, listing, qa, quality).
        metrics: Flat ``{metric_name: value}`` used for MLflow and the drift check.
        tables: Named DataFrames or dicts rendered in the scorecard.
        provenance: What was recomputed, from which inputs, and how many LLM calls were made.
        seconds: Wall-clock time of the evaluation.
    """

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    tables: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0


def git_info() -> dict[str, Any]:
    """Commit hash, branch and whether tracked files have uncommitted changes."""

    def run(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        return out.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "short": run("rev-parse", "--short", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def load_script(name: str) -> ModuleType:
    """Import ``scripts/<name>.py`` as a module so its functions can be reused."""
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"catalogiq_script_{name}", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_script(name: str, args: list[str] | None = None) -> float:
    """Run ``scripts/<name>.py`` in a subprocess (log kept in the eval logs folder).

    Returns:
        Wall-clock seconds. Raises ``RuntimeError`` with the log tail if the script fails.
    """
    logs = resolve_path(get_eval_config()["paths"]["logs_dir"])
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}.log"
    start = time.perf_counter()
    with log.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / f"{name}.py"), *(args or [])],
            cwd=PROJECT_ROOT,
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if proc.returncode != 0:
        tail = log.read_text(encoding="utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"scripts/{name}.py failed (exit {proc.returncode}):\n{tail}")
    return time.perf_counter() - start


class CountingClient(LLMClient):
    """Counts the calls that reach the model (cache misses when placed under a CachedClient)."""

    def __init__(self, inner: LLMClient, max_calls: int | None = None) -> None:
        self.inner = inner
        self.model = inner.model
        self.calls = 0
        self.max_calls = max_calls

    def chat(self, *args: Any, **kwargs: Any) -> LLMResponse:
        """Forward to the wrapped client and count the call; stop when the budget is used up."""
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise RuntimeError(
                f"LLM cache miss budget of {self.max_calls} calls used up: the cache does not "
                "cover this evaluation. Re-run with --allow-llm-calls N to let the model run "
                "(slow on CPU)."
            )
        self.calls += 1
        return self.inner.chat(*args, **kwargs)


LLM_CALL_BUDGET: dict[str, int | None] = {"max_calls": 25}


def counted_cached_client() -> tuple[LLMClient, CountingClient]:
    """The project's disk-cached LLM client plus a counter of the calls that missed the cache.

    At most ``LLM_CALL_BUDGET["max_calls"]`` misses are allowed (``None`` = unlimited).
    """
    from catalogiq.config import get_rag_config
    from catalogiq.llm.client import CachedClient, get_llm_client

    counter = CountingClient(get_llm_client(cache=False), LLM_CALL_BUDGET["max_calls"])
    cache_dir = resolve_path(get_rag_config()["paths"]["llm_cache_dir"])
    return CachedClient(counter, cache_dir), counter


def table_to_markdown(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Render a DataFrame as a GitHub markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append("" if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def results_dir() -> Path:
    """Folder for the Phase 7 outputs (created on demand)."""
    path = resolve_path(get_eval_config()["paths"]["out_dir"])
    path.mkdir(parents=True, exist_ok=True)
    return path
