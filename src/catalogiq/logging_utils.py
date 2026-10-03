"""Standard logger setup for CatalogIQ."""

from __future__ import annotations

import logging
import sys

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str, level: str | int = "INFO") -> logging.Logger:
    """Return a logger that writes to stdout with a consistent format.

    Args:
        name: Logger name, usually ``__name__``.
        level: Logging level name or number.

    Returns:
        A configured ``logging.Logger`` (handlers are added only once).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
