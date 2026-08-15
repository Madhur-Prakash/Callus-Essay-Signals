"""Essay metadata and privacy endpoints.

The privacy endpoint exists so the frontend can state, accurately and from the
live configuration, whether a submitted essay will be stored. Hard-coding that
claim in the UI would let it drift out of step with the server's actual setting -
which is the one thing a privacy notice must never do.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Query

from app.api.dependencies import DbDep, SettingsDep
from app.core.exceptions import AnalysisNotFoundError, PersistenceUnavailableError
from app.schemas.essay import EssayListResponse, PrivacyInfoResponse

router = APIRouter(prefix="/essays", tags=["essays"])


DERIVED_DATA_STORED = (
    "analysis id, timestamps, detector and model versions",
    "the verdict, calibrated probabilities and confidence band",
    "aggregate statistics (word counts, perplexity, burstiness, and so on)",
    "per-sentence character offsets, scores and classifications",
    "the deterministic evidence generated for flagged sentences",
)
TEXT_ITEMS = ("the essay text", "the text of individual sentences")


@router.get(
    "/privacy",
    response_model=PrivacyInfoResponse,
    summary="What this server stores",
)
async def privacy(settings: SettingsDep):  # noqa: ANN201
    stored = list(DERIVED_DATA_STORED)
    never_stored: list[str] = []
    if settings.save_essays:
        stored.append("the essay text and per-sentence text (SAVE_ESSAYS is enabled)")
    else:
        never_stored.extend(TEXT_ITEMS)

    return PrivacyInfoResponse(
        save_essays_default=settings.save_essays,
        per_request_override_supported=True,
        retention_days=settings.analysis_retention_days,
        what_is_stored=stored,
        what_is_never_stored=never_stored,
        what_is_never_logged=[
            "the essay text, in full or in part",
            "the text of individual sentences",
            "email addresses, phone numbers or other personal identifiers "
            "(scrubbed defensively before any log write)",
        ],
        deletion_endpoint="DELETE /api/v1/analysis/{analysis_id}",
    )


@router.get("", response_model=EssayListResponse, summary="List stored essay metadata")
async def list_essays(db: DbDep, limit: int = Query(default=20, ge=1, le=100)):  # noqa: ANN201
    if not db.available:
        raise PersistenceUnavailableError()
    rows = await db.list_essays(limit=limit)
    return EssayListResponse(
        total=len(rows),
        items=[
            {
                "essay_id": row["essay_id"],
                "content_hash": row.get("content_hash", ""),
                "n_words": row.get("n_words", 0),
                "n_characters": row.get("n_characters", 0),
                "n_paragraphs": row.get("n_paragraphs", 0),
                "created_at": _iso(row.get("created_at")),
                "text_stored": bool(row.get("text_stored")),
                "analysis_ids": [],
            }
            for row in rows
        ],
    )


@router.get("/{essay_id}", summary="Fetch stored essay metadata")
async def get_essay(essay_id: str, db: DbDep):  # noqa: ANN201
    if not db.available:
        raise PersistenceUnavailableError()
    row = await db.get_essay(essay_id)
    if row is None:
        raise AnalysisNotFoundError("No essay exists with that identifier.")
    row.pop("text", None)  # never returned over the API
    row["created_at"] = _iso(row.get("created_at"))
    return row


def _iso(value) -> str | None:  # noqa: ANN001
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
