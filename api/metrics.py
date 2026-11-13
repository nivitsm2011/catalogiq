"""In-memory request metrics (counts and latencies per endpoint)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque

import numpy as np

KEEP = 500  # latencies kept per endpoint for the p95


class RequestMetrics:
    """Thread-safe request counters."""

    def __init__(self) -> None:
        self.started = time.time()
        self._lock = threading.Lock()
        self._count: dict[str, int] = defaultdict(int)
        self._errors: dict[str, int] = defaultdict(int)
        self._lat: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=KEEP))
        self._sum: dict[str, float] = defaultdict(float)

    def record(self, endpoint: str, status: int, ms: float) -> None:
        """Record one finished request."""
        with self._lock:
            self._count[endpoint] += 1
            self._sum[endpoint] += ms
            self._lat[endpoint].append(ms)
            if status >= 400:
                self._errors[endpoint] += 1

    def snapshot(self) -> dict:
        """Totals and per-endpoint count, errors, average and p95 latency (milliseconds)."""
        with self._lock:
            endpoints = {
                name: {
                    "count": n,
                    "errors": self._errors[name],
                    "avg_ms": self._sum[name] / n,
                    "p95_ms": float(np.percentile(list(self._lat[name]), 95)),
                }
                for name, n in self._count.items()
            }
            total = sum(self._count.values())
            return {
                "total_requests": total,
                "total_errors": sum(self._errors.values()),
                "avg_latency_ms": (sum(self._sum.values()) / total) if total else 0.0,
                "uptime_s": time.time() - self.started,
                "endpoints": endpoints,
            }
