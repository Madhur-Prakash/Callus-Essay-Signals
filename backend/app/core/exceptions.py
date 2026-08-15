"""Application error types and the FastAPI handlers that render them.

Users never see a Python traceback: every handler returns a small JSON envelope
with a machine-readable ``code`` and a human-readable ``message``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, scrub

logger = get_logger("app.errors")

# Numeric literals rather than `starlette.status` constants: several of those
# names were renamed across Starlette versions (422 and 413 in particular), and
# the wire format is what actually matters here.
HTTP_400_BAD_REQUEST = 400
HTTP_404_NOT_FOUND = 404
HTTP_413_PAYLOAD_TOO_LARGE = 413
HTTP_422_UNPROCESSABLE = 422
HTTP_429_TOO_MANY_REQUESTS = 429
HTTP_500_INTERNAL_ERROR = 500
HTTP_503_SERVICE_UNAVAILABLE = 503
HTTP_504_GATEWAY_TIMEOUT = 504


class AppError(Exception):
    """Base class for expected, user-facing failures."""

    status_code: int = HTTP_400_BAD_REQUEST
    code: str = "app_error"
    message: str = "The request could not be completed."

    def __init__(self, message: str | None = None, **details: Any) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        self.details = details


class EssayTooShortError(AppError):
    status_code = HTTP_422_UNPROCESSABLE
    code = "essay_too_short"
    message = "The essay is too short to analyse reliably."


class EssayTooLongError(AppError):
    status_code = HTTP_413_PAYLOAD_TOO_LARGE
    code = "essay_too_long"
    message = "The essay exceeds the maximum supported length."


class EmptyEssayError(AppError):
    status_code = HTTP_422_UNPROCESSABLE
    code = "essay_empty"
    message = "No essay text was provided."


class ModelNotTrainedError(AppError):
    status_code = HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_trained"
    message = (
        "The detector model has not been trained yet. "
        "Run the training pipeline (`uv run python -m ml.training.train`) first."
    )


class ModelUnavailableError(AppError):
    status_code = HTTP_503_SERVICE_UNAVAILABLE
    code = "model_unavailable"
    message = "The language model could not be loaded."


class AnalysisNotFoundError(AppError):
    status_code = HTTP_404_NOT_FOUND
    code = "analysis_not_found"
    message = "No analysis exists with that identifier."


class PersistenceUnavailableError(AppError):
    status_code = HTTP_503_SERVICE_UNAVAILABLE
    code = "persistence_unavailable"
    message = "Persistent storage is unavailable, so this analysis cannot be retrieved."


class EvaluationNotAvailableError(AppError):
    status_code = HTTP_404_NOT_FOUND
    code = "evaluation_unavailable"
    message = (
        "No evaluation report has been generated yet. "
        "Run `uv run python -m ml.evaluation.evaluate`."
    )


class RateLimitExceededError(AppError):
    status_code = HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limit_exceeded"
    message = "Too many analysis requests. Please wait before trying again."


class AnalysisTimeoutError(AppError):
    status_code = HTTP_504_GATEWAY_TIMEOUT
    code = "analysis_timeout"
    message = "The analysis took too long and was cancelled."


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    """Attach all handlers to the application."""

    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        logger.warning(f"app_error | code={exc.code} status={exc.status_code}")
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.code, exc.message, exc.details or None),
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [
            {"field": ".".join(str(p) for p in err.get("loc", ())[1:]), "issue": err.get("msg", "")}
            for err in exc.errors()
        ]
        logger.warning(f"validation_error | fields={len(fields)}")
        return JSONResponse(
            status_code=HTTP_422_UNPROCESSABLE,
            content=_envelope(
                "validation_error",
                "The request body did not pass validation.",
                {"fields": fields[:10]},
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(f"http_{exc.status_code}", str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Log the type and a scrubbed message; never return internals to callers.
        logger.error(
            f"unhandled_exception | path={request.url.path} type={type(exc).__name__} "
            f"detail={scrub(str(exc))[:300]}",
            exc_info=True,
        )
        return JSONResponse(
            status_code=HTTP_500_INTERNAL_ERROR,
            content=_envelope(
                "internal_error",
                "An unexpected error occurred. The incident has been logged.",
            ),
        )
