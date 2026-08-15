"""Redis cache - used only where it measurably helps.

Two justified uses in this system:

1. **Analysis result cache.** A full analysis costs one language-model forward
   pass per 512-token window plus a spaCy parse; on a 1,200-word essay that is
   roughly 1-3 seconds of CPU. The pipeline is deterministic for a fixed
   ``(essay, detector_version, model_version)`` triple, so the result is exactly
   cacheable under ``SHA256(essay + detector_version + model_version)``.
   Re-analysing the same draft - which users do constantly while editing - then
   costs a single round trip.
2. **Distributed rate limiting** (see :mod:`app.core.rate_limit`), which needs
   shared state to be correct across workers.

Redis is *disabled by default*. Nothing degrades without it beyond losing the
cache; every call site treats a ``None`` client as "no cache".
"""

from __future__ import annotations

import json
from typing import Any

import redis.asyncio as aioredis

from app.config import Settings, get_settings
from app.core.logging import get_logger, log_event

logger = get_logger("app.redis")

_client: aioredis.Redis | None = None
_available = False
_last_error: str | None = None

CACHE_PREFIX = "analysis:v1:"
META_PREFIX = "modelmeta:v1:"


async def connect_redis(settings: Settings | None = None) -> bool:
    """Create the Redis connection pool. Returns False when unavailable."""
    global _client, _available, _last_error
    settings = settings or get_settings()
    if not settings.redis_enabled:
        log_event(logger, "redis.disabled", reason="REDIS_ENABLED=false")
        return False
    try:
        _client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        await _client.ping()
        _available = True
        _last_error = None
        log_event(logger, "redis.connected", ttl_seconds=settings.redis_cache_ttl_seconds)
        return True
    except Exception as exc:
        _available = False
        _last_error = type(exc).__name__
        _client = None
        log_event(
            logger,
            "redis.connect_failed",
            level="warning",
            type=type(exc).__name__,
            hint="caching disabled for this process",
        )
        return False


async def close_redis() -> None:
    global _client, _available
    if _client is not None:
        await _client.aclose()
        _client = None
        _available = False
        log_event(logger, "redis.closed")


async def get_redis() -> aioredis.Redis | None:
    return _client if _available else None


def redis_status() -> dict[str, Any]:
    settings = get_settings()
    return {
        "enabled": settings.redis_enabled,
        "available": _available,
        "last_error": _last_error,
    }


async def ping_redis() -> bool:
    global _available
    if _client is None:
        return False
    try:
        await _client.ping()
        _available = True
        return True
    except Exception:
        _available = False
        return False


# --------------------------------------------------------------------------- #
# Analysis result cache
# --------------------------------------------------------------------------- #
async def cache_get_analysis(content_hash: str) -> dict[str, Any] | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(CACHE_PREFIX + content_hash)
    except Exception as exc:  # pragma: no cover - network dependent
        log_event(logger, "redis.get_failed", level="warning", type=type(exc).__name__)
        return None
    if raw is None:
        log_event(logger, "cache.miss", key_digest=content_hash[:12])
        return None
    log_event(logger, "cache.hit", key_digest=content_hash[:12])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


async def cache_set_analysis(content_hash: str, payload: dict[str, Any]) -> None:
    client = await get_redis()
    if client is None:
        return
    settings = get_settings()
    try:
        await client.set(
            CACHE_PREFIX + content_hash,
            json.dumps(payload, default=str),
            ex=settings.redis_cache_ttl_seconds,
        )
        log_event(logger, "cache.store", key_digest=content_hash[:12])
    except Exception as exc:  # pragma: no cover
        log_event(logger, "redis.set_failed", level="warning", type=type(exc).__name__)


async def cache_invalidate_all() -> int:
    """Drop every cached analysis (used when a new model is activated)."""
    client = await get_redis()
    if client is None:
        return 0
    deleted = 0
    try:
        async for key in client.scan_iter(match=CACHE_PREFIX + "*", count=500):
            await client.delete(key)
            deleted += 1
    except Exception as exc:  # pragma: no cover
        log_event(logger, "redis.invalidate_failed", level="warning", type=type(exc).__name__)
    if deleted:
        log_event(logger, "cache.invalidated", keys=deleted)
    return deleted


# --------------------------------------------------------------------------- #
# Model metadata cache (avoids re-reading artifact JSON on every /model/info hit)
# --------------------------------------------------------------------------- #
async def cache_get_model_meta(key: str) -> dict[str, Any] | None:
    client = await get_redis()
    if client is None:
        return None
    try:
        raw = await client.get(META_PREFIX + key)
        return json.loads(raw) if raw else None
    except Exception:
        return None


async def cache_set_model_meta(key: str, payload: dict[str, Any]) -> None:
    client = await get_redis()
    if client is None:
        return
    try:
        await client.set(META_PREFIX + key, json.dumps(payload, default=str), ex=3600)
    except Exception:  # pragma: no cover
        pass
