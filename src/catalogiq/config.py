"""Configuration loading for CatalogIQ.

All paths, seeds and model names live in ``configs/*.yaml``; code reads them here.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: Config file path. Defaults to ``configs/base.yaml``.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


LLM_HOST_ENV = "CATALOGIQ_LLM_HOST"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the cached default configuration.

    The environment variable ``CATALOGIQ_LLM_HOST`` overrides ``models.llm.host`` (the Docker
    compose file points it at the ``ollama`` container); nothing else is read from the environment.
    """
    cfg = load_config()
    host = os.environ.get(LLM_HOST_ENV)
    if host:
        cfg["models"]["llm"]["host"] = host
    return cfg


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


@cache
def get_vision_config() -> dict[str, Any]:
    """Return the ``vision`` section of ``configs/vision.yaml`` (Phase 2 classifier settings)."""
    return load_config(PROJECT_ROOT / "configs" / "vision.yaml")["vision"]


@cache
def get_search_config() -> dict[str, Any]:
    """Return the ``search`` section of ``configs/search.yaml`` (Phase 3 settings)."""
    return load_config(PROJECT_ROOT / "configs" / "search.yaml")["search"]


@cache
def get_rag_config() -> dict[str, Any]:
    """Return the ``rag`` section of ``configs/rag.yaml`` (Phase 4 settings)."""
    return load_config(PROJECT_ROOT / "configs" / "rag.yaml")["rag"]


@cache
def get_qa_config() -> dict[str, Any]:
    """Return the ``qa`` section of ``configs/qa.yaml`` (Phase 5 settings)."""
    return load_config(PROJECT_ROOT / "configs" / "qa.yaml")["qa"]


@cache
def get_quality_config() -> dict[str, Any]:
    """Return the ``quality`` section of ``configs/quality.yaml`` (Phase 6 settings)."""
    return load_config(PROJECT_ROOT / "configs" / "quality.yaml")["quality"]


@cache
def get_eval_config() -> dict[str, Any]:
    """Return the ``eval`` section of ``configs/eval.yaml`` (Phase 7 settings)."""
    return load_config(PROJECT_ROOT / "configs" / "eval.yaml")["eval"]


@cache
def get_api_config() -> dict[str, Any]:
    """Return the ``api`` section of ``configs/api.yaml`` (limits, upload rules, startup files)."""
    return load_config(PROJECT_ROOT / "configs" / "api.yaml")["api"]
