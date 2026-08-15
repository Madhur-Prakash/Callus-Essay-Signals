"""MongoDB document builders for analyses and essays.

The important logic here is privacy shaping. Two documents are written:

``analyses``          verdict, summary, evidence, timings - no essay text, and no
                      sentence text unless ``SAVE_ESSAYS`` is on.
``analysis_results``  per-sentence and per-paragraph rows. Sentences are always
                      stored as ``(start, end, score, classification)`` offsets;
                      the text itself is included only when persistence is
                      enabled, so a stored analysis of a non-saved essay cannot
                      be used to reconstruct the essay.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from app.services.detector import AnalysisResult


def _now() -> datetime:
    return datetime.now(UTC)


def essay_id_for(content_hash: str) -> str:
    return hashlib.sha256(content_hash.encode("utf-8")).hexdigest()[:24]


def build_essay_document(
    result: AnalysisResult, *, store_text: bool, text: str | None = None
) -> dict[str, Any]:
    summary = result.summary
    document: dict[str, Any] = {
        "essay_id": essay_id_for(result.content_hash),
        "content_hash": result.content_hash,
        "n_words": summary["n_words"],
        "n_characters": summary["n_characters"],
        "n_sentences": summary["n_sentences"],
        "n_paragraphs": summary["n_paragraphs"],
        "created_at": _now(),
        "text_stored": bool(store_text and text),
    }
    if store_text and text:
        document["text"] = text
    return document


def build_analysis_document(
    result: AnalysisResult, *, store_text: bool
) -> dict[str, Any]:
    """The ``analyses`` row: everything except the text."""
    return {
        "analysis_id": result.analysis_id,
        "essay_id": essay_id_for(result.content_hash),
        "content_hash": result.content_hash,
        "status": result.status,
        "created_at": _now(),
        "updated_at": _now(),
        "classification": result.verdict.get("classification"),
        "label": result.verdict.get("label"),
        "confidence": result.verdict.get("confidence"),
        "confidence_score": result.verdict.get("confidence_score"),
        "probabilities": result.verdict.get("probabilities"),
        "margin": result.verdict.get("margin"),
        "abstained": result.verdict.get("abstained"),
        "abstain_reason": result.verdict.get("abstain_reason"),
        "summary": result.summary,
        "evidence": result.evidence,
        "repetition": result.repetition,
        "model": result.model_info,
        "timings": result.timings,
        "warnings": result.warnings,
        "text_stored": bool(store_text),
        "detector_version": result.model_info.get("detector_version"),
        "model_version": result.model_info.get("model_version"),
    }


def build_results_document(
    result: AnalysisResult, *, store_text: bool
) -> dict[str, Any]:
    """The ``analysis_results`` row: per-sentence and per-paragraph detail."""
    sentences = []
    for sentence in result.sentences:
        row = {
            "sentence_id": sentence["sentence_id"],
            "paragraph_id": sentence["paragraph_id"],
            "start": sentence["start"],
            "end": sentence["end"],
            "score": sentence["score"],
            "classification": sentence["classification"],
            "confidence": sentence["confidence"],
            "n_words": sentence["n_words"],
            "features": sentence.get("features", {}),
        }
        if "evidence" in sentence:
            row["evidence"] = sentence["evidence"]
        if store_text:
            row["text"] = sentence["text"]
        sentences.append(row)

    return {
        "analysis_id": result.analysis_id,
        "created_at": _now(),
        "text_stored": bool(store_text),
        "paragraphs": result.paragraphs,
        "sentences": sentences,
        "rhythm": result.rhythm,
    }


def rehydrate_response(
    analysis: dict[str, Any], results: dict[str, Any] | None
) -> dict[str, Any]:
    """Rebuild an API response shape from stored documents.

    Sentence ``text`` is empty when the essay was not persisted. The frontend
    handles that by slicing the essay it still has in the editor using the stored
    offsets, which is why offsets are always kept.
    """
    payload: dict[str, Any] = {
        "analysis_id": analysis["analysis_id"],
        "status": analysis.get("status", "completed"),
        "classification": analysis.get("classification", "insufficient_evidence"),
        "label": analysis.get("label", ""),
        "description": analysis.get("description", ""),
        "confidence": analysis.get("confidence", "none"),
        "confidence_score": analysis.get("confidence_score", 0.0),
        "probabilities": analysis.get("probabilities", {}),
        "margin": analysis.get("margin", 0.0),
        "abstained": analysis.get("abstained", False),
        "abstain_reason": analysis.get("abstain_reason"),
        "summary": analysis.get("summary", {}),
        "evidence": analysis.get("evidence", {}),
        "repetition": analysis.get("repetition", {}),
        "model": analysis.get("model", {}),
        "timings": analysis.get("timings", {}),
        "content_hash": analysis.get("content_hash", ""),
        "created_at": _iso(analysis.get("created_at")),
        "persisted": True,
        "cached": False,
        "warnings": analysis.get("warnings", []),
        "paragraphs": (results or {}).get("paragraphs", []),
        "sentences": [],
        "rhythm": (results or {}).get("rhythm", []),
    }
    for sentence in (results or {}).get("sentences", []):
        payload["sentences"].append({**sentence, "text": sentence.get("text", "")})
    return payload


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)
