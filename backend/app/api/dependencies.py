"""FastAPI dependencies: settings, database handles, rate limiting, request ids."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, Request

from app.config import Settings, get_settings
from app.core.exceptions import RateLimitExceededError
from app.core.rate_limit import RateLimiter
from app.db.mongodb import MongoManager, mongo
from app.services.detector import Detector, detector

_limiter: RateLimiter | None = None


def get_rate_limiter(settings: Settings = Depends(get_settings)) -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(settings)
    return _limiter


def client_key(request: Request) -> str:
    """Identify the caller for rate limiting.

    Uses the first entry of ``X-Forwarded-For`` when present (so the limit is per
    real client behind a proxy) and falls back to the socket address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


async def enforce_rate_limit(
    request: Request,
    limiter: RateLimiter = Depends(get_rate_limiter),
) -> None:
    allowed, retry_after = await limiter.check(client_key(request))
    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit reached. Try again in about {retry_after} seconds.",
            retry_after_seconds=retry_after,
        )


def get_request_id(
    x_request_id: Annotated[str | None, Header(alias="X-Request-ID")] = None,
) -> str:
    """Propagate a caller-supplied request id, or mint one."""
    return x_request_id or uuid.uuid4().hex[:16]


def get_detector() -> Detector:
    return detector


def get_db() -> MongoManager:
    return mongo


SettingsDep = Annotated[Settings, Depends(get_settings)]
DetectorDep = Annotated[Detector, Depends(get_detector)]
DbDep = Annotated[MongoManager, Depends(get_db)]
RequestIdDep = Annotated[str, Depends(get_request_id)]
