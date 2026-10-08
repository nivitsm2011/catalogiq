"""Configuration loading for CatalogIQ.

All paths, seeds and model names live in ``configs/*.yaml``; code reads them here.
"""

from __future__ import annotations

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


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Return the cached default configuration."""
    return load_config()


def resolve_path(relative: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    path = Path(relative)
    return path if path.is_absolute() else PROJECT_ROOT / path


@cache
def get_vision_config() -> dict[str, Any]:
    """Return the ``vision`` section of ``configs/vision.yaml`` (Phase 2 classifier settings)."""
    return load_config(PROJECT_ROOT / "configs" / "vision.yaml")["vision"]
