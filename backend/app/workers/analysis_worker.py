"""Kafka consumer that runs analyses out of band.

Only needed when ``KAFKA_ENABLED=true``. Run it as a separate process:

    uv run python -m app.workers.analysis_worker

Flow:

    essay.analysis.requests --> this worker --> detector pipeline --> MongoDB
                                                         |
                                                         +--> essay.analysis.results

The heavy model work is dispatched to a thread so the consumer's event loop keeps
sending heartbeats; without that a long analysis would look like a dead consumer
and the broker would rebalance the partition away mid-job.
"""

from __future__ import annotations

import asyncio
import json
import signal
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger, log_duration, log_event, safe_text_meta
from app.db.kafka import publish_result_event
from app.db.mongodb import mongo
from app.models.analysis import (
    build_analysis_document,
    build_essay_document,
    build_results_document,
)
from app.services.detector import detector

logger = get_logger("app.worker", file="worker.log")

_shutdown = asyncio.Event()


async def handle_message(payload: dict[str, Any]) -> None:
    settings = get_settings()
    analysis_id = payload.get("analysis_id") or "unknown"
    text = payload.get("text") or ""
    save = bool(payload.get("save")) and settings.save_essays

    if not text.strip():
        log_event(logger, "worker.empty_payload", level="warning", analysis_id=analysis_id)
        if mongo.available:
            await mongo.update_analysis_status(
                analysis_id, "failed", error="empty payload"
            )
        return

    log_event(
        logger,
        "worker.received",
        analysis_id=analysis_id,
        **safe_text_meta(text, prefix="essay"),
    )
    if mongo.available:
        await mongo.update_analysis_status(analysis_id, "processing")

    try:
        with log_duration(logger, "worker.analysis", analysis_id=analysis_id):
            # to_thread keeps the consumer heartbeating during the CPU-bound run.
            result = await asyncio.to_thread(detector.analyse, text, analysis_id=analysis_id)
    except Exception as exc:
        log_event(
            logger,
            "worker.analysis_failed",
            level="error",
            analysis_id=analysis_id,
            type=type(exc).__name__,
        )
        if mongo.available:
            await mongo.update_analysis_status(
                analysis_id, "failed", error=type(exc).__name__
            )
        await publish_result_event(analysis_id=analysis_id, status="failed")
        return

    if mongo.available:
        await mongo.insert_essay(
            build_essay_document(result, store_text=save, text=text if save else None)
        )
        await mongo.insert_analysis(
            build_analysis_document(result, store_text=save),
            build_results_document(result, store_text=save),
        )
        result.persisted = True

    await publish_result_event(
        analysis_id=analysis_id,
        status="completed",
        classification=result.verdict.get("classification"),
    )
    log_event(
        logger,
        "worker.completed",
        analysis_id=analysis_id,
        classification=result.verdict.get("classification"),
        persisted=result.persisted,
    )


async def run() -> None:
    settings = get_settings()
    if not settings.kafka_enabled:
        log_event(
            logger,
            "worker.kafka_disabled",
            level="warning",
            hint="set KAFKA_ENABLED=true to use the async path",
        )
        return

    from aiokafka import AIOKafkaConsumer

    await mongo.connect()
    detector.load()
    detector.warmup()

    consumer = AIOKafkaConsumer(
        settings.kafka_analysis_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_consumer_group,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_interval_ms=600_000,
    )
    await consumer.start()
    log_event(
        logger,
        "worker.started",
        topic=settings.kafka_analysis_topic,
        group=settings.kafka_consumer_group,
    )

    try:
        while not _shutdown.is_set():
            batch = await consumer.getmany(timeout_ms=1000, max_records=1)
            for _partition, messages in batch.items():
                for message in messages:
                    try:
                        await handle_message(message.value)
                    except Exception as exc:  # pragma: no cover
                        log_event(
                            logger,
                            "worker.message_failed",
                            level="error",
                            type=type(exc).__name__,
                        )
                    finally:
                        # Commit either way: a poison message must not block the
                        # partition forever. The failure is recorded in MongoDB.
                        await consumer.commit()
    finally:
        await consumer.stop()
        await mongo.close()
        log_event(logger, "worker.stopped")


def main() -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _request_shutdown(*_: Any) -> None:
        log_event(logger, "worker.shutdown_requested")
        loop.call_soon_threadsafe(_shutdown.set)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _request_shutdown)
        except (ValueError, AttributeError):  # pragma: no cover - Windows subset
            pass

    try:
        loop.run_until_complete(run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
