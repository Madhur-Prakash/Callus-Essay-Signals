"""Infrastructure-layer tests: MongoDB, Redis cache, Kafka gating, rate limiting."""

from __future__ import annotations

import pytest

from app.core.logging import content_hash, safe_text_meta, scrub, text_digest
from app.db.kafka import kafka_status, should_queue
from app.db.mongodb import MongoManager
from app.db.redis import cache_get_analysis, cache_set_analysis, redis_status


class TestPrivacyHelpers:
    def test_safe_text_meta_never_includes_the_text(self) -> None:
        text = "My name is Ada and I built a robot."
        meta = safe_text_meta(text)
        serialised = str(meta)
        assert "Ada" not in serialised
        assert "robot" not in serialised
        assert meta["text_chars"] == len(text)
        assert meta["text_words"] == len(text.split())

    def test_digest_is_stable_and_non_reversible(self) -> None:
        assert text_digest("hello") == text_digest("hello")
        assert text_digest("hello") != text_digest("hellp")
        assert len(text_digest("hello")) == 16

    def test_content_hash_includes_the_versions(self) -> None:
        a = content_hash("essay", detector_version="1.0.0", model_version="1.0.0")
        b = content_hash("essay", detector_version="1.0.0", model_version="1.0.1")
        c = content_hash("essay", detector_version="1.0.1", model_version="1.0.0")
        assert len({a, b, c}) == 3, "a version change must invalidate the cache key"

    def test_scrub_removes_emails_phones_and_ssns(self) -> None:
        cleaned = scrub("Contact ada@example.com or +1 415 555 0134, ssn 123-45-6789")
        assert "ada@example.com" not in cleaned
        assert "555" not in cleaned
        assert "123-45-6789" not in cleaned
        assert "<email>" in cleaned and "<phone>" in cleaned

    def test_scrub_leaves_ordinary_prose_alone(self) -> None:
        text = "The servo burned out twice that winter."
        assert scrub(text) == text


class TestRedisGating:
    def test_status_reports_disabled_rather_than_broken(self) -> None:
        status = redis_status()
        assert status["enabled"] is False
        assert status["available"] is False

    async def test_cache_reads_and_writes_are_no_ops_when_disabled(self) -> None:
        await cache_set_analysis("abc", {"analysis_id": "x"})
        assert await cache_get_analysis("abc") is None


class TestKafkaGating:
    def test_status_reports_disabled(self) -> None:
        status = kafka_status()
        assert status["enabled"] is False
        assert status["available"] is False
        assert status["async_threshold_chars"] > 0

    def test_nothing_is_queued_while_kafka_is_disabled(self) -> None:
        """Even a huge essay and an explicit request must run inline when the
        broker is not connected — otherwise the request would hang forever."""
        assert should_queue(1_000_000, requested=True) is False
        assert should_queue(10, requested=False) is False


class TestRateLimiter:
    async def test_allows_up_to_the_limit_then_rejects(self, settings) -> None:  # noqa: ANN001
        from app.config import Settings
        from app.core.rate_limit import RateLimiter

        limited = Settings(
            rate_limit_enabled=True,
            rate_limit_requests=3,
            rate_limit_window_seconds=60,
            redis_enabled=False,
        )
        limiter = RateLimiter(limited)
        for _ in range(3):
            allowed, _ = await limiter.check("client-a")
            assert allowed is True

        allowed, retry_after = await limiter.check("client-a")
        assert allowed is False
        assert retry_after > 0

    async def test_limits_are_per_client(self) -> None:
        from app.config import Settings
        from app.core.rate_limit import RateLimiter

        limiter = RateLimiter(
            Settings(
                rate_limit_enabled=True,
                rate_limit_requests=1,
                rate_limit_window_seconds=60,
                redis_enabled=False,
            )
        )
        assert (await limiter.check("client-a"))[0] is True
        assert (await limiter.check("client-a"))[0] is False
        assert (await limiter.check("client-b"))[0] is True

    async def test_disabled_limiter_always_allows(self) -> None:
        from app.config import Settings
        from app.core.rate_limit import RateLimiter

        limiter = RateLimiter(Settings(rate_limit_enabled=False, redis_enabled=False))
        for _ in range(50):
            assert (await limiter.check("anyone"))[0] is True


@pytest.mark.integration
class TestMongoDb:
    async def test_connect_and_roundtrip_an_analysis(self) -> None:
        manager = MongoManager()
        if not await manager.connect():
            pytest.skip("MongoDB is not running")

        try:
            document = {
                "analysis_id": "test-analysis-roundtrip",
                "status": "completed",
                "classification": "human",
                "content_hash": "hash-roundtrip",
                "summary": {"n_words": 100},
            }
            results = {
                "analysis_id": "test-analysis-roundtrip",
                "sentences": [{"sentence_id": 0, "start": 0, "end": 5, "score": 0.1}],
                "paragraphs": [],
            }
            await manager.insert_analysis(document, results)

            fetched = await manager.get_analysis("test-analysis-roundtrip")
            assert fetched is not None
            assert fetched["classification"] == "human"
            assert "_id" not in fetched, "the Mongo _id must not leak into responses"

            rows = await manager.get_analysis_results("test-analysis-roundtrip")
            assert rows is not None and len(rows["sentences"]) == 1

            by_hash = await manager.find_analysis_by_hash("hash-roundtrip")
            assert by_hash is not None

            assert await manager.delete_analysis("test-analysis-roundtrip") is True
            assert await manager.get_analysis("test-analysis-roundtrip") is None
        finally:
            await manager.close()

    async def test_unavailable_mongo_degrades_instead_of_raising(self) -> None:
        from app.config import Settings

        manager = MongoManager(
            Settings(
                mongodb_url="mongodb://127.0.0.1:1",  # nothing listens here
                mongodb_timeout_ms=250,
                mongodb_enabled=True,
            )
        )
        assert await manager.connect() is False
        assert manager.available is False
        assert manager.last_error is not None
        await manager.close()
