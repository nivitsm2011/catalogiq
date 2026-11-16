"""Request-size cap and a small in-process rate limiter (no extra dependency)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from catalogiq.config import get_api_config


class RateLimiter:
    """Sliding-window counter per (client, bucket).

    Args:
        window_s: Window length in seconds.
        clock: Time source (tests pass a fake clock).
    """

    def __init__(self, window_s: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.window_s = window_s
        self.clock = clock
        self._hits: dict[tuple[str, str], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, bucket: str, limit: int) -> tuple[bool, float]:
        """Record a request; returns ``(allowed, seconds_until_a_slot_frees)``."""
        now = self.clock()
        with self._lock:
            hits = self._hits[(client, bucket)]
            while hits and now - hits[0] >= self.window_s:
                hits.popleft()
            if len(hits) >= limit:
                return False, max(0.0, self.window_s - (now - hits[0]))
            hits.append(now)
            return True, 0.0


def _error(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> Response:
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message}},
        headers=headers,
    )


def make_limit_middleware(cfg: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Build the ASGI middleware: 413 for oversized bodies, 429 when a client is too fast."""
    limits = (cfg or get_api_config())["limits"]
    rate = limits["rate_limit"]
    max_body = int(limits["max_body_bytes"])
    limiter = RateLimiter(float(rate["window_s"]))
    heavy = tuple(rate["heavy_paths"])
    exempt = tuple(rate["exempt_paths"])

    async def middleware(request: Request, call_next: Callable[..., Any]) -> Response:
        declared = request.headers.get("content-length")
        if declared is not None:
            try:
                too_big = int(declared) > max_body
            except ValueError:
                return _error(400, "bad_content_length", "The Content-Length header is invalid.")
            if too_big:
                mb = max_body / 1024 / 1024
                return _error(
                    413,
                    "request_too_large",
                    f"The request is larger than the {mb:.1f} MB limit (photos can be up to 5 MB).",
                )
        path = request.url.path
        if rate["enabled"] and not path.startswith(exempt):
            client = request.client.host if request.client else "unknown"
            is_heavy = request.method != "OPTIONS" and path.startswith(heavy)
            bucket, limit = (
                ("heavy", int(rate["heavy_max_requests"]))
                if is_heavy
                else ("all", int(rate["max_requests"]))
            )
            ok, wait = limiter.allow(client, bucket, limit)
            if not ok:
                retry = max(1, int(wait + 0.999))
                return _error(
                    429,
                    "rate_limited",
                    f"Too many requests; wait about {retry} s and try again.",
                    {"Retry-After": str(retry)},
                )
        return await call_next(request)

    return middleware
