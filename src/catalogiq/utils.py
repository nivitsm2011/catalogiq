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
