"""Small shared helpers: reproducible seeding and timing."""

from __future__ import annotations

import functools
import os
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from catalogiq.logging_utils import get_logger

logger = get_logger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        logger.debug("torch not installed; skipped torch seeding")


def configure_cache_dirs() -> None:
    """Keep pretrained-weight caches inside the project folder (``paths.cache``).

    Must run before ``torch.hub``/``open_clip``/``huggingface_hub`` download anything.
    """
    from catalogiq.config import get_config, resolve_path

    cache = resolve_path(get_config()["paths"]["cache"])
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def get_device() -> str:
    """Return ``"cuda"`` if a GPU is available, otherwise ``"cpu"``."""
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def timed(func: F) -> F:
    """Decorator that logs how long a function took."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            logger.info("%s took %.2fs", func.__qualname__, time.perf_counter() - start)

    return wrapper  # type: ignore[return-value]
