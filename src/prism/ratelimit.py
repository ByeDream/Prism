"""In-memory sliding-window rate limiter (per-client, per-minute)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Thread-safe sliding-window rate limiter.

    Each client has a deque of request timestamps.  ``check()`` prunes
    entries older than 60 seconds, then allows or denies the request
    based on the remaining count vs. the client's configured limit.
    """

    def __init__(self) -> None:
        self._windows: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, client_name: str, limit: int) -> bool:
        """Return True if the request is allowed, False if rate-limited.

        A *limit* of 0 means unlimited (always allowed).
        """
        if limit <= 0:
            return True

        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            window = self._windows[client_name]
            while window and window[0] <= cutoff:
                window.popleft()
            if len(window) >= limit:
                return False
            window.append(now)
            return True
