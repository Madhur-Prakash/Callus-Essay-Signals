"""FastAPI application factory and lifespan.

Startup order matters: the language model and spaCy pipeline are loaded once
during startup (a model load costs ~10-20 s on CPU; doing it lazily would make
the first user's request look broken), while MongoDB, Redis and Kafka are all
optional and their absence only degrades specific capabilities.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import analysis, essays, evaluation, health, model
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger, log_event
from app.db.kafka import close_kafka, connect_kafka
from app.db.mongodb import mongo
from app.db.redis import close_redis, connect_redis
from app.services.detector import detector

logger = get_logger("app.main")

DESCRIPTION = """
Explainable detection of AI-generated and AI-polished writing in college
admissions essays.

**This is not an LLM wrapper.** A small local causal language model
(`distilgpt2` by default) is used as a measuring instrument to obtain per-token
log probabilities, entropy and rank. Those numbers, together with stylometric,
syntactic, burstiness, repetition, within-document style-shift and reference-corpus
features, form a ~411-dimensional document vector. The classification is produced
by our own trained, calibrated scikit-learn classifier. No hosted chat model is
consulted during analysis.

**Every flag carries evidence.** Explanations are generated deterministically from
measured feature values — percentile position within the human training
distribution, deviation from the essay's own baseline, and the classifier's own
per-feature contributions.

**Detection is not proof of authorship.** See `/api/v1/model/info` for the
methodology and documented limitations, and `/api/v1/evaluation` for held-out
metrics including the bias analysis.
"""

TAGS_METADATA = [
    {"name": "analysis", "description": "Submit essays and retrieve results."},
    {"name": "health", "description": "Per-component health and readiness."},
    {"name": "model", "description": "Active model version and methodology."},
    {
        "name": "evaluation",
        "description": "Held-out metrics, failure analysis and the dataset card.",
    },
    {"name": "essays", "description": "Stored essay metadata and the privacy policy."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging()
    started = time.perf_counter()

    log_event(
        logger,
        "app.starting",
        environment=settings.app_env,
        detector_version=settings.detector_version,
        save_essays=settings.save_essays,
        mongodb=settings.mongodb_enabled,
        redis=settings.redis_enabled,
        kafka=settings.kafka_enabled,
    )

    await mongo.connect()
    await connect_redis(settings)
    await connect_kafka(settings)

    # Load artifacts, then force the heavy models in so request one is fast.
    status = detector.load()
    if not settings.lm_lazy_load:
        try:
            detector.warmup()
        except Exception as exc:  # noqa: BLE001 - startup must survive this
            log_event(
                logger,
                "app.warmup_failed",
                level="error",
                type=type(exc).__name__,
                hint="the model will be loaded on the first request instead",
            )

    if not status["models_ready"]:
        log_event(
            logger,
            "app.model_not_trained",
            level="warning",
            hint=(
                "run `uv run python -m ml.training.prepare_dataset`, "
                "`... extract_features`, then `... train`"
            ),
        )

    # Retention housekeeping on boot: cheap, and keeps a long-running dev
    # instance from holding old analyses indefinitely.
    if mongo.available and settings.analysis_retention_days > 0:
        try:
            await mongo.purge_expired()
        except Exception as exc:  # noqa: BLE001
            log_event(logger, "app.purge_failed", level="warning", type=type(exc).__name__)

    log_event(
        logger,
        "app.started",
        startup_ms=round((time.perf_counter() - started) * 1000, 2),
        models_ready=status["models_ready"],
    )
    try:
        yield
    finally:
        log_event(logger, "app.stopping")
        await close_kafka()
        await close_redis()
        await mongo.close()
        from logifyx import flush

        flush(timeout=3.0)
        log_event(logger, "app.stopped")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="AI Essay Detector API",
        description=DESCRIPTION,
        version=settings.detector_version,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        max_age=600,
    )

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):  # noqa: ANN001, ANN202
        """Reject oversized bodies before they are buffered or parsed."""
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > settings.max_request_bytes:
                log_event(
                    logger,
                    "api.body_too_large",
                    level="warning",
                    bytes=int(content_length),
                    limit=settings.max_request_bytes,
                )
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "request_too_large",
                            "message": (
                                f"Request body exceeds the "
                                f"{settings.max_request_bytes:,} byte limit."
                            ),
                        }
                    },
                )
        return await call_next(request)

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):  # noqa: ANN001, ANN202
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        # Path and status only — never query strings or bodies, which could carry
        # essay content.
        if request.url.path.startswith(settings.api_v1_prefix):
            log_event(
                logger,
                "api.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(elapsed_ms, 2),
            )
        return response

    register_exception_handlers(app)

    prefix = settings.api_v1_prefix
    app.include_router(health.router, prefix=prefix)
    app.include_router(analysis.router, prefix=prefix)
    app.include_router(model.router, prefix=prefix)
    app.include_router(evaluation.router, prefix=prefix)
    app.include_router(essays.router, prefix=prefix)

    @app.get("/", include_in_schema=False)
    async def root():  # noqa: ANN202
        return {
            "name": "AI Essay Detector API",
            "version": settings.detector_version,
            "docs": "/docs",
            "health": f"{prefix}/health",
            "analyse": f"POST {prefix}/analysis",
            "note": (
                "The classification is made by our own trained classifier. The local "
                "language model provides token probabilities only; no hosted chat "
                "model is consulted."
            ),
        }

    return app


app = create_app()
