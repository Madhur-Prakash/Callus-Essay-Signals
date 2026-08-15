"""Model information endpoint.

Exposes the active model version and, importantly, *what the system is and is
not*: the methodology block below is served to the frontend's "How this works"
panel so the explanation the user reads comes from the backend rather than being
duplicated (and eventually contradicted) in the UI code.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import SettingsDep
from app.db.redis import cache_get_model_meta, cache_set_model_meta
from app.schemas.analysis import ModelInfoResponse
from app.services.classifier import detector_models

router = APIRouter(prefix="/model", tags=["model"])

METHODOLOGY = {
    "summary": (
        "This detector does not ask another AI whether an essay is AI-written. It "
        "measures properties of the writing and feeds those measurements to a "
        "classifier we trained ourselves."
    ),
    "pipeline": [
        "Normalise the text and split it into paragraphs and sentences",
        "Measure stylometry: sentence and word statistics, punctuation, vocabulary",
        "Parse part-of-speech and dependency structure with spaCy",
        "Score every token with a small local language model to get log "
        "probabilities, entropy and rank",
        "Measure sentence rhythm (burstiness) and n-gram / syntactic repetition",
        "Compare each sentence against the author's own baseline (style shift)",
        "Compare the text against the human and machine reference corpora",
        "Feed ~411 document features into our trained classifier",
        "Calibrate the probability and map it to a confidence band",
        "Generate evidence deterministically from the measured feature values",
    ],
    "what_the_language_model_does": (
        "It provides one number per token: how probable that token was given the "
        "preceding text. It is a measuring instrument, like a thermometer. It is "
        "never asked to judge authorship, and no hosted chat model is called at "
        "any point during analysis."
    ),
    "what_makes_the_decision": (
        "A scikit-learn classifier trained on our labelled corpus, with document-"
        "level grouped train/validation/test splits and probability calibration."
    ),
    "signals_measured": [
        "token predictability (log probability, perplexity, entropy, rank)",
        "sentence rhythm and length variation (burstiness)",
        "vocabulary richness and lexical diversity",
        "punctuation and function-word habits",
        "part-of-speech and dependency distributions",
        "repeated words, phrases and syntactic templates",
        "shifts in style within the essay",
        "similarity to human and machine reference corpora",
    ],
    "limitations": [
        "Detection is probabilistic. A flag is evidence for review, never proof of "
        "authorship.",
        "Low perplexity on its own is not evidence of machine authorship — clear, "
        "conventional human prose also scores low.",
        "Uniform sentence length on its own is not evidence either; many people "
        "write evenly.",
        "AI detectors are known to over-flag writing by people who learned English "
        "as an additional language. See the evaluation report's bias section.",
        "A lightly edited human essay is mostly human text, so the AI-polished class "
        "is inherently the hardest to identify.",
        "The detector cannot establish who wrote a text and must not be the sole "
        "basis for any decision about a person.",
    ],
}


@router.get("/info", response_model=ModelInfoResponse, summary="Active model information")
async def model_info(settings: SettingsDep):  # noqa: ANN201
    cache_key = f"{settings.detector_version}:{detector_models.metadata.get('model_version')}"
    cached = await cache_get_model_meta(cache_key)
    if cached is not None:
        return cached

    if not detector_models.ready:
        detector_models.load()

    payload = {
        **detector_models.info(),
        "methodology": METHODOLOGY,
        "analysis_thresholds": _thresholds(settings),
    }
    await cache_set_model_meta(cache_key, payload)
    return payload


def _thresholds(settings) -> dict[str, object]:  # noqa: ANN001
    """The limits the client needs in order to set expectations honestly.

    Two different gates, and conflating them produces a misleading UI:

    * ``min_chars`` / ``max_chars`` are **hard** — outside them the request is
      rejected with 4xx.
    * ``min_sentences_for_verdict`` / ``min_words_for_verdict`` are **soft** — the
      request succeeds, but the detector declines to name a class because the
      distributional measurements are not stable on so little text.

    Served from configuration so the interface cannot drift out of step with what
    the server actually enforces.
    """
    from app.services.calibration import (
        MIN_SENTENCES_FOR_VERDICT,
        MIN_WORDS_FOR_VERDICT,
    )

    return {
        "min_chars": settings.min_essay_chars,
        "max_chars": settings.max_essay_chars,
        "min_sentences_for_verdict": MIN_SENTENCES_FOR_VERDICT,
        "min_words_for_verdict": MIN_WORDS_FOR_VERDICT,
        "note": (
            "Text shorter than min_chars is rejected. Text that clears min_chars but "
            "has fewer than min_sentences_for_verdict sentences or "
            "min_words_for_verdict words is analysed, but the verdict will be "
            "'insufficient_evidence'."
        ),
    }


@router.post(
    "/reload",
    summary="Reload model artifacts from disk",
    description=(
        "Picks up a newly trained model without restarting the process, and clears "
        "the analysis cache so results are not served from a previous model version."
    ),
)
async def reload_model():  # noqa: ANN201
    from app.db.redis import cache_invalidate_all
    from app.services.detector import detector

    status = detector.load()
    invalidated = await cache_invalidate_all()
    return {
        "reloaded": detector_models.ready,
        "model_version": detector_models.metadata.get("model_version"),
        "error": detector_models.load_error,
        "cache_entries_invalidated": invalidated,
        "status": status,
    }
