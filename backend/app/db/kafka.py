"""Kafka producer - disabled by default, and for good reason.

A 250-word essay takes ~0.3 s to analyse and a 1,200-word essay ~1.7 s. Putting
that behind a message queue would add a broker round trip, a polling loop in the
frontend and two extra failure modes in exchange for nothing. So the default path
is synchronous.

Kafka earns its place for work that genuinely does not fit a request:

* essays above ``ASYNC_THRESHOLD_CHARS`` (25,000 characters - roughly 4,000
  words, where analysis moves into tens of seconds)
* batch dataset generation and corpus processing
* evaluation and training runs triggered from the API

The topology, when enabled:

    FastAPI --> essay.analysis.requests --> analysis_worker --> MongoDB
                                                    |
                                                    +--> essay.analysis.results

The frontend polls ``GET /api/v1/analysis/{id}`` for status. Everything here is
written so that enabling Kafka is a config change, not a code change.
"""

from __future__ import annotations

import json
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import get_logger, log_event

logger = get_logger("app.kafka")

_producer: Any = None
_available = False
_last_error: str | None = None


async def connect_kafka(settings: Settings | None = None) -> bool:
    global _producer, _available, _last_error
    settings = settings or get_settings()
    if not settings.kafka_enabled:
        log_event(logger, "kafka.disabled", reason="KAFKA_ENABLED=false")
        return False
    try:
        from aiokafka import AIOKafkaProducer

        _producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda v: v.encode("utf-8") if isinstance(v, str) else v,
            request_timeout_ms=10_000,
            linger_ms=20,
        )
        await _producer.start()
        _available = True
        _last_error = None
        log_event(
            logger,
            "kafka.connected",
            servers=settings.kafka_bootstrap_servers,
            topic=settings.kafka_analysis_topic,
        )
        return True
    except Exception as exc:
        _available = False
        _last_error = type(exc).__name__
        _producer = None
        log_event(
            logger,
            "kafka.connect_failed",
            level="warning",
            type=type(exc).__name__,
            hint="analysis will run synchronously",
        )
        return False


async def close_kafka() -> None:
    global _producer, _available
    if _producer is not None:
        try:
            await _producer.stop()
        except Exception:  # pragma: no cover
            pass
        _producer = None
        _available = False
        log_event(logger, "kafka.closed")


def kafka_status() -> dict[str, Any]:
    settings = get_settings()
    return {
        "enabled": settings.kafka_enabled,
        "available": _available,
        "last_error": _last_error,
        "request_topic": settings.kafka_analysis_topic,
        "result_topic": settings.kafka_result_topic,
        "async_threshold_chars": settings.async_threshold_chars,
    }


def should_queue(text_length: int, *, requested: bool) -> bool:
    """Whether this request should go through the queue.

    True only when Kafka is actually connected AND either the caller asked for it
    or the essay is large enough that inline analysis would hold a request open
    for an unreasonable time.
    """
    settings = get_settings()
    if not (settings.kafka_enabled and _available):
        return False
    return requested or text_length >= settings.async_threshold_chars


async def publish_analysis_request(
    *, analysis_id: str, text: str, content_hash: str, save: bool
) -> bool:
    """Enqueue an analysis job. Returns False if it could not be published."""
    settings = get_settings()
    if _producer is None or not _available:
        return False
    try:
        await _producer.send_and_wait(
            settings.kafka_analysis_topic,
            key=analysis_id,
            value={
                "analysis_id": analysis_id,
                "text": text,
                "content_hash": content_hash,
                "save": save,
                "detector_version": settings.detector_version,
            },
        )
        # Note the absence of the essay text in this log line.
        log_event(
            logger,
            "kafka.request_published",
            analysis_id=analysis_id,
            topic=settings.kafka_analysis_topic,
            payload_chars=len(text),
        )
        return True
    except Exception as exc:
        log_event(
            logger,
            "kafka.publish_failed",
            level="error",
            analysis_id=analysis_id,
            type=type(exc).__name__,
        )
        return False


async def publish_result_event(
    *, analysis_id: str, status: str, classification: str | None = None
) -> None:
    """Emit a completion event. Fire-and-forget: never fail an analysis over it."""
    settings = get_settings()
    if _producer is None or not _available:
        return
    try:
        await _producer.send_and_wait(
            settings.kafka_result_topic,
            key=analysis_id,
            value={
                "analysis_id": analysis_id,
                "status": status,
                "classification": classification,
            },
        )
        log_event(logger, "kafka.result_published", analysis_id=analysis_id, status=status)
    except Exception as exc:  # pragma: no cover
        log_event(
            logger, "kafka.result_publish_failed", level="warning", type=type(exc).__name__
        )
