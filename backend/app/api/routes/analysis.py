"""Analysis endpoints."""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Query, Request

from app.api.dependencies import DbDep, DetectorDep, RequestIdDep, SettingsDep, enforce_rate_limit
from app.core.exceptions import (
    AnalysisNotFoundError,
    AnalysisTimeoutError,
    PersistenceUnavailableError,
)
from app.core.logging import (
    content_hash as compute_hash,
)
from app.core.logging import (
    get_logger,
    log_event,
    safe_text_meta,
    with_context,
)
from app.core.security import validate_essay
from app.db import kafka as kafka_module
from app.db.redis import cache_get_analysis, cache_set_analysis
from app.models.analysis import (
    build_analysis_document,
    build_essay_document,
    build_results_document,
    rehydrate_response,
)
from app.schemas.analysis import (
    AnalysisListResponse,
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatusResponse,
    DeleteResponse,
    QueuedAnalysisResponse,
    SentenceListResponse,
)

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = get_logger("app.api.analysis")

ANALYSIS_TIMEOUT_SECONDS = 120


@router.post(
    "",
    response_model=AnalysisResponse,
    response_model_exclude_none=False,
    responses={
        202: {"model": QueuedAnalysisResponse, "description": "Queued for async processing"},
        413: {"description": "Essay too long"},
        422: {"description": "Essay empty or too short"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Model not trained or unavailable"},
    },
    summary="Analyse an essay",
    description=(
        "Runs the full detection pipeline and returns a calibrated verdict with "
        "sentence-level scores and measured evidence. The classification is produced "
        "by our own trained classifier; the language model contributes token "
        "probabilities only."
    ),
    dependencies=[Depends(enforce_rate_limit)],
)
async def create_analysis(
    payload: AnalysisRequest,
    request: Request,
    settings: SettingsDep,
    detector: DetectorDep,
    db: DbDep,
    request_id: RequestIdDep,
):  # noqa: ANN201 - response model declared above
    log = with_context(logger, request_id=request_id)
    text = validate_essay(payload.text, settings)

    # Persistence: the server setting is the ceiling. A request may opt out of
    # storage but can never opt in to something the operator disabled.
    store_text = bool(settings.save_essays and (payload.save is not False))

    model_version = str(detector_model_version(detector))
    digest = compute_hash(
        text, detector_version=settings.detector_version, model_version=model_version
    )
    log_event(log, "api.analysis_request", **safe_text_meta(text, prefix="essay"))

    # ---------------------------------------------------------------- cache
    cached = await cache_get_analysis(digest)
    if cached is not None:
        cached["cached"] = True
        log_event(log, "api.analysis_cache_hit", analysis_id=cached.get("analysis_id"))
        return cached

    # ------------------------------------------------------------ async path
    if kafka_module.should_queue(len(text), requested=payload.async_mode):
        analysis_id = uuid.uuid4().hex
        published = await kafka_module.publish_analysis_request(
            analysis_id=analysis_id, text=text, content_hash=digest, save=store_text
        )
        if published:
            if db.available:
                await db.update_analysis_status(
                    analysis_id,
                    "queued",
                    content_hash=digest,
                    created_at=datetime.now(UTC),
                )
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=202,
                content=QueuedAnalysisResponse(
                    analysis_id=analysis_id,
                    status="queued",
                    poll_url=f"{settings.api_v1_prefix}/analysis/{analysis_id}",
                    message=(
                        "This essay was queued for background analysis. Poll the "
                        "returned URL for its status."
                    ),
                    content_hash=digest,
                    created_at=datetime.now(UTC).isoformat(),
                ).model_dump(),
            )
        log_event(log, "api.queue_failed_running_inline", level="warning")

    # ------------------------------------------------------- synchronous path
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(detector.analyse, text),
            timeout=ANALYSIS_TIMEOUT_SECONDS,
        )
    except TimeoutError as exc:
        log_event(log, "api.analysis_timeout", level="error", limit_s=ANALYSIS_TIMEOUT_SECONDS)
        raise AnalysisTimeoutError(
            f"Analysis exceeded the {ANALYSIS_TIMEOUT_SECONDS}s limit. "
            "Try a shorter essay, or enable the asynchronous path."
        ) from exc

    # ---------------------------------------------------------- persistence
    if db.available:
        try:
            await db.insert_essay(
                build_essay_document(result, store_text=store_text, text=text)
            )
            await db.insert_analysis(
                build_analysis_document(result, store_text=store_text),
                build_results_document(result, store_text=store_text),
            )
            result.persisted = True
        except Exception as exc:  # noqa: BLE001 - persistence must not fail analysis
            log_event(
                log,
                "api.persist_failed",
                level="warning",
                analysis_id=result.analysis_id,
                type=type(exc).__name__,
            )

    response = result.to_dict()
    await cache_set_analysis(digest, response)
    return response


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse | AnalysisStatusResponse,
    summary="Fetch a stored analysis",
)
async def get_analysis(analysis_id: str, db: DbDep, request_id: RequestIdDep):  # noqa: ANN201
    log = with_context(logger, request_id=request_id, analysis_id=analysis_id)
    if not db.available:
        raise PersistenceUnavailableError()

    analysis = await db.get_analysis(analysis_id)
    if analysis is None:
        raise AnalysisNotFoundError()

    status = analysis.get("status", "completed")
    if status != "completed":
        log_event(log, "api.analysis_status", status=status)
        return AnalysisStatusResponse(
            analysis_id=analysis_id,
            status=status,
            created_at=_iso(analysis.get("created_at")),
            updated_at=_iso(analysis.get("updated_at")),
            classification=analysis.get("classification"),
            confidence=analysis.get("confidence"),
            error=analysis.get("error"),
        )

    results = await db.get_analysis_results(analysis_id)
    return rehydrate_response(analysis, results)


@router.get(
    "/{analysis_id}/sentences",
    response_model=SentenceListResponse,
    summary="Per-sentence results for an analysis",
)
async def get_analysis_sentences(analysis_id: str, db: DbDep):  # noqa: ANN201
    if not db.available:
        raise PersistenceUnavailableError()
    analysis = await db.get_analysis(analysis_id)
    if analysis is None:
        raise AnalysisNotFoundError()
    results = await db.get_analysis_results(analysis_id)
    sentences = (results or {}).get("sentences", [])
    return SentenceListResponse(
        analysis_id=analysis_id,
        n_sentences=len(sentences),
        sentences=[{**s, "text": s.get("text", "")} for s in sentences],
    )


@router.get("", response_model=AnalysisListResponse, summary="List recent analyses")
async def list_analyses(
    db: DbDep,
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
):  # noqa: ANN201
    if not db.available:
        raise PersistenceUnavailableError()
    rows = await db.list_analyses(limit=limit, skip=skip)
    total = await db.count_analyses()
    return AnalysisListResponse(
        total=total,
        items=[
            {
                "analysis_id": row["analysis_id"],
                "status": row.get("status", "completed"),
                "classification": row.get("classification"),
                "confidence": row.get("confidence"),
                "created_at": _iso(row.get("created_at")),
                "n_words": (row.get("summary") or {}).get("n_words"),
                "n_sentences": (row.get("summary") or {}).get("n_sentences"),
                "flagged_sentences": (row.get("summary") or {}).get("flagged_sentences"),
            }
            for row in rows
        ],
    )


@router.delete(
    "/{analysis_id}",
    response_model=DeleteResponse,
    summary="Delete an analysis",
    description=(
        "Removes the analysis and its per-sentence results. Provided so a user can "
        "withdraw a submission after the fact."
    ),
)
async def delete_analysis(analysis_id: str, db: DbDep, request_id: RequestIdDep):  # noqa: ANN201
    if not db.available:
        raise PersistenceUnavailableError()
    deleted = await db.delete_analysis(analysis_id)
    log_event(
        with_context(logger, request_id=request_id),
        "api.analysis_deleted",
        analysis_id=analysis_id,
        deleted=deleted,
    )
    if not deleted:
        raise AnalysisNotFoundError()
    return DeleteResponse(analysis_id=analysis_id, deleted=True)


def detector_model_version(detector) -> str:  # noqa: ANN001
    from app.services.classifier import detector_models

    return str(detector_models.metadata.get("model_version", "untrained"))


def _iso(value) -> str | None:  # noqa: ANN001
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
