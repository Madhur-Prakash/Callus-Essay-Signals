"""MongoDB access layer (Motor / async).

The application treats persistence as *optional but preferred*: if MongoDB is
unreachable the API still analyses essays and returns results, it just cannot
serve them again later. That behaviour is surfaced honestly through
``/api/v1/health`` and in the analysis response ``persisted`` flag.

Collections
-----------
``essays``           essay metadata (+ text only when ``SAVE_ESSAYS=true``)
``analyses``         one document per analysis: verdict, summary, evidence, timings
``analysis_results`` per-sentence and per-paragraph rows for an analysis
``model_versions``   registry of trained detector artifacts
``evaluation_runs``  stored evaluation reports (metrics, bias, failures)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

from app.config import Settings, get_settings
from app.core.logging import get_logger, log_event

logger = get_logger("app.mongodb")

ESSAYS = "essays"
ANALYSES = "analyses"
ANALYSIS_RESULTS = "analysis_results"
MODEL_VERSIONS = "model_versions"
EVALUATION_RUNS = "evaluation_runs"


class MongoManager:
    """Owns the Motor client lifecycle and all query helpers."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: AsyncIOMotorClient | None = None
        self._db: AsyncIOMotorDatabase | None = None
        self.available = False
        self.last_error: str | None = None

    # ------------------------------------------------------------ lifecycle
    async def connect(self) -> bool:
        settings = self._settings
        if not settings.mongodb_enabled:
            log_event(logger, "mongodb.disabled", level="warning")
            return False
        try:
            self._client = AsyncIOMotorClient(
                settings.mongodb_url,
                serverSelectionTimeoutMS=settings.mongodb_timeout_ms,
                connectTimeoutMS=settings.mongodb_timeout_ms,
                uuidRepresentation="standard",
                appname=settings.app_name,
            )
            info = await self._client.admin.command("ping")
            self._db = self._client[settings.mongodb_database]
            self.available = bool(info.get("ok"))
            self.last_error = None
            await self._ensure_indexes()
            log_event(
                logger,
                "mongodb.connected",
                database=settings.mongodb_database,
            )
            return True
        except Exception as exc:
            self.available = False
            self.last_error = type(exc).__name__
            log_event(
                logger,
                "mongodb.connect_failed",
                level="warning",
                type=type(exc).__name__,
                hint="analysis will run without persistence",
            )
            return False

    async def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
            self._db = None
            self.available = False
            log_event(logger, "mongodb.closed")

    @property
    def db(self) -> AsyncIOMotorDatabase:
        if self._db is None:
            raise RuntimeError("MongoDB is not connected")
        return self._db

    async def ping(self) -> bool:
        if self._client is None:
            return False
        try:
            await self._client.admin.command("ping")
            self.available = True
            return True
        except PyMongoError:
            self.available = False
            return False

    async def _ensure_indexes(self) -> None:
        db = self.db
        await db[ANALYSES].create_index([("analysis_id", ASCENDING)], unique=True)
        await db[ANALYSES].create_index([("created_at", DESCENDING)])
        await db[ANALYSES].create_index([("content_hash", ASCENDING)])
        await db[ANALYSES].create_index([("status", ASCENDING)])
        await db[ANALYSIS_RESULTS].create_index([("analysis_id", ASCENDING)], unique=True)
        await db[ESSAYS].create_index([("essay_id", ASCENDING)], unique=True)
        await db[ESSAYS].create_index([("created_at", DESCENDING)])
        await db[MODEL_VERSIONS].create_index(
            [("model_version", ASCENDING), ("detector_version", ASCENDING)], unique=True
        )
        await db[EVALUATION_RUNS].create_index([("created_at", DESCENDING)])
        await db[EVALUATION_RUNS].create_index([("run_id", ASCENDING)], unique=True)
        log_event(logger, "mongodb.indexes_ready", collections=5)

    # --------------------------------------------------------------- essays
    async def insert_essay(self, document: dict[str, Any]) -> None:
        await self.db[ESSAYS].update_one(
            {"essay_id": document["essay_id"]}, {"$setOnInsert": document}, upsert=True
        )

    async def get_essay(self, essay_id: str) -> dict[str, Any] | None:
        return await self.db[ESSAYS].find_one({"essay_id": essay_id}, {"_id": 0})

    async def list_essays(self, limit: int = 20) -> list[dict[str, Any]]:
        cursor = self.db[ESSAYS].find({}, {"_id": 0, "text": 0}).sort("created_at", DESCENDING)
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------------- analyses
    async def insert_analysis(
        self, analysis: dict[str, Any], results: dict[str, Any] | None = None
    ) -> None:
        await self.db[ANALYSES].update_one(
            {"analysis_id": analysis["analysis_id"]}, {"$set": analysis}, upsert=True
        )
        if results is not None:
            await self.db[ANALYSIS_RESULTS].update_one(
                {"analysis_id": analysis["analysis_id"]}, {"$set": results}, upsert=True
            )

    async def update_analysis_status(
        self, analysis_id: str, status: str, **fields: Any
    ) -> None:
        await self.db[ANALYSES].update_one(
            {"analysis_id": analysis_id},
            {"$set": {"status": status, "updated_at": _utcnow(), **fields}},
            upsert=True,
        )

    async def get_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        return await self.db[ANALYSES].find_one({"analysis_id": analysis_id}, {"_id": 0})

    async def get_analysis_results(self, analysis_id: str) -> dict[str, Any] | None:
        return await self.db[ANALYSIS_RESULTS].find_one({"analysis_id": analysis_id}, {"_id": 0})

    async def find_analysis_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        return await self.db[ANALYSES].find_one(
            {"content_hash": content_hash, "status": "completed"},
            {"_id": 0},
            sort=[("created_at", DESCENDING)],
        )

    async def list_analyses(self, limit: int = 20, skip: int = 0) -> list[dict[str, Any]]:
        cursor = (
            self.db[ANALYSES]
            .find({}, {"_id": 0, "evidence": 0})
            .sort("created_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return await cursor.to_list(length=limit)

    async def count_analyses(self) -> int:
        return await self.db[ANALYSES].count_documents({})

    async def delete_analysis(self, analysis_id: str) -> bool:
        res = await self.db[ANALYSES].delete_one({"analysis_id": analysis_id})
        await self.db[ANALYSIS_RESULTS].delete_one({"analysis_id": analysis_id})
        return res.deleted_count > 0

    async def purge_expired(self) -> int:
        """Delete analyses older than the configured retention window."""
        days = self._settings.analysis_retention_days
        if days <= 0:
            return 0
        cutoff = _utcnow() - timedelta(days=days)
        stale = self.db[ANALYSES].find({"created_at": {"$lt": cutoff}}, {"analysis_id": 1})
        ids = [doc["analysis_id"] async for doc in stale]
        if not ids:
            return 0
        await self.db[ANALYSES].delete_many({"analysis_id": {"$in": ids}})
        await self.db[ANALYSIS_RESULTS].delete_many({"analysis_id": {"$in": ids}})
        await self.db[ESSAYS].delete_many({"created_at": {"$lt": cutoff}})
        log_event(logger, "mongodb.retention_purge", deleted=len(ids), retention_days=days)
        return len(ids)

    # ------------------------------------------------------- model registry
    async def register_model_version(self, metadata: dict[str, Any]) -> None:
        await self.db[MODEL_VERSIONS].update_one(
            {
                "model_version": metadata.get("model_version"),
                "detector_version": metadata.get("detector_version"),
            },
            {"$set": {**metadata, "registered_at": _utcnow()}},
            upsert=True,
        )

    async def list_model_versions(self, limit: int = 10) -> list[dict[str, Any]]:
        cursor = self.db[MODEL_VERSIONS].find({}, {"_id": 0}).sort("registered_at", DESCENDING)
        return await cursor.to_list(length=limit)

    # ------------------------------------------------------ evaluation runs
    async def store_evaluation_run(self, report: dict[str, Any]) -> None:
        await self.db[EVALUATION_RUNS].update_one(
            {"run_id": report["run_id"]},
            {"$set": {**report, "stored_at": _utcnow()}},
            upsert=True,
        )

    async def latest_evaluation_run(self) -> dict[str, Any] | None:
        return await self.db[EVALUATION_RUNS].find_one(
            {}, {"_id": 0}, sort=[("created_at", DESCENDING)]
        )


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Module-level singleton, wired up in the application lifespan.
mongo = MongoManager()


async def get_mongo() -> MongoManager:
    return mongo
