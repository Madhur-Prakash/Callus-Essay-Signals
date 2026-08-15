"""Fixed-window rate limiting.

Uses Redis when it is enabled (so limits hold across worker processes) and falls
back to an in-process counter otherwise. The fallback is honest about its scope:
it protects a single Uvicorn worker, which is the right amount of protection for
local development.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

from app.config import Settings
from app.core.logging import get_logger

logger = get_logger("app.rate_limit")


class RateLimiter:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    async def check(self, key: str) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        settings = self._settings
        if not settings.rate_limit_enabled:
            return True, 0

        window = settings.rate_limit_window_seconds
        limit = settings.rate_limit_requests

        if settings.redis_enabled:
            allowed = await self._check_redis(key, limit, window)
            if allowed is not None:
                return allowed, 0 if allowed else window
            # Redis unreachable: degrade to the local counter rather than
            # rejecting or silently allowing everything.

        return self._check_local(key, limit, window)

    async def _check_redis(self, key: str, limit: int, window: int) -> bool | None:
        from app.db.redis import get_redis

        client = await get_redis()
        if client is None:
            return None
        try:
            redis_key = f"ratelimit:{key}:{int(time.time()) // window}"
            pipe = client.pipeline()
            pipe.incr(redis_key)
            pipe.expire(redis_key, window + 1)
            count, _ = await pipe.execute()
            return int(count) <= limit
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(f"rate_limit.redis_error | type={type(exc).__name__}")
            return None

    def _check_local(self, key: str, limit: int, window: int) -> tuple[bool, int]:
        now = time.monotonic()
        bucket = self._hits[key]
        cutoff = now - window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()
        if len(bucket) >= limit:
            retry_after = max(1, int(window - (now - bucket[0])))
            return False, retry_after
        bucket.append(now)
        if len(self._hits) > 10_000:  # crude guard against unbounded growth
            self._hits = defaultdict(
                deque, {k: v for k, v in self._hits.items() if v and v[-1] > cutoff}
            )
        return True, 0
