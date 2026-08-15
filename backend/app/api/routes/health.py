"""Health and readiness endpoints.

Health reports *per component* rather than a single boolean, because this system
deliberately degrades: MongoDB down means "cannot save", Redis down means "no
cache", spaCy missing means "reduced features". A flat 200/500 would hide all of
that.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

from fastapi import APIRouter

from app.api.dependencies import DbDep, DetectorDep, SettingsDep
from app.db.kafka import kafka_status
from app.db.redis import ping_redis, redis_status
from app.schemas.analysis import ComponentHealth, HealthResponse
from app.services.classifier import detector_models

router = APIRouter(tags=["health"])
_STARTED_AT = time.monotonic()


@router.get("/health", response_model=HealthResponse, summary="Component health")
async def health(settings: SettingsDep, db: DbDep, detector: DetectorDep):  # noqa: ANN201
    mongo_ok = await db.ping() if settings.mongodb_enabled else False
    redis_ok = await ping_redis() if settings.redis_enabled else False
    redis = redis_status()
    kafka = kafka_status()
    status_info = detector.status()

    components = [
        ComponentHealth(
            name="mongodb",
            enabled=settings.mongodb_enabled,
            available=mongo_ok,
            detail=(
                f"database={settings.mongodb_database}"
                if mongo_ok
                else db.last_error or "unreachable; analyses run but are not persisted"
            ),
        ),
        ComponentHealth(
            name="redis",
            enabled=redis["enabled"],
            available=redis_ok,
            detail=(
                f"analysis cache active (ttl {settings.redis_cache_ttl_seconds}s)"
                if redis_ok
                else "disabled by configuration"
                if not redis["enabled"]
                else redis["last_error"] or "unreachable; caching disabled"
            ),
        ),
        ComponentHealth(
            name="kafka",
            enabled=kafka["enabled"],
            available=kafka["available"],
            detail=(
                f"async path active above {kafka['async_threshold_chars']} chars"
                if kafka["available"]
                else "disabled by configuration; analyses run synchronously"
                if not kafka["enabled"]
                else kafka["last_error"] or "unreachable"
            ),
        ),
        ComponentHealth(
            name="language_model",
            enabled=True,
            available=bool(status_info["language_model"]["loaded"]),
            detail=(
                f"{settings.lm_model_name} on {settings.lm_device} "
                f"(instrument only, does not classify)"
            ),
        ),
        ComponentHealth(
            name="spacy",
            enabled=True,
            available=bool(status_info["spacy"]["loaded"]),
            detail=status_info["spacy"].get("error")
            or f"{settings.spacy_model} ({status_info['spacy'].get('backend')})",
        ),
        ComponentHealth(
            name="detector_model",
            enabled=True,
            available=detector_models.ready,
            detail=(
                f"model_version={detector_models.metadata.get('model_version')} "
                f"regime={detector_models.metadata.get('data_regime')}"
                if detector_models.ready
                else detector_models.load_error or "not trained"
            ),
        ),
        ComponentHealth(
            name="corpus_reference",
            enabled=True,
            available=bool(status_info["corpus_reference"].get("fitted")),
            detail="cross-corpus similarity features"
            if status_info["corpus_reference"].get("fitted")
            else "not fitted; cor_* features are zero",
        ),
    ]

    # The detector model is the only component whose absence makes the service
    # unable to do its job at all.
    if not detector_models.ready:
        overall = "unavailable"
    elif not all(c.available for c in components if c.enabled):
        overall = "degraded"
    else:
        overall = "ok"

    return HealthResponse(
        status=overall,
        version=settings.detector_version,
        environment=settings.app_env,
        components=components,
        detector=status_info,
        uptime_seconds=round(time.monotonic() - _STARTED_AT, 2),
        checked_at=datetime.now(UTC).isoformat(),
    )


@router.get("/health/live", summary="Liveness probe")
async def live():  # noqa: ANN201
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe")
async def ready(detector: DetectorDep):  # noqa: ANN201
    """Ready only when the detector can actually answer a request."""
    from fastapi.responses import JSONResponse

    is_ready = detector_models.ready
    return JSONResponse(
        status_code=200 if is_ready else 503,
        content={
            "ready": is_ready,
            "reason": None if is_ready else (detector_models.load_error or "model not trained"),
        },
    )
