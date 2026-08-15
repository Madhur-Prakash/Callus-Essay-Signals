"""Schemas for the essays resource."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EssayMetadata(BaseModel):
    essay_id: str
    content_hash: str
    n_words: int
    n_characters: int
    n_paragraphs: int
    created_at: str
    text_stored: bool = Field(
        description=(
            "Whether the essay body itself was persisted. False means only derived "
            "metrics and character offsets were kept."
        )
    )
    analysis_ids: list[str] = Field(default_factory=list)


class EssayListResponse(BaseModel):
    total: int
    items: list[EssayMetadata]


class PrivacyInfoResponse(BaseModel):
    save_essays_default: bool
    per_request_override_supported: bool
    retention_days: int
    what_is_stored: list[str]
    what_is_never_stored: list[str]
    what_is_never_logged: list[str]
    deletion_endpoint: str
