"""Probability calibration and the confidence language shown to users.

Two separate jobs:

1. **Calibration fitting** (training time) — Platt scaling / isotonic regression
   over a held-out split, so that "0.8" means roughly "80% of documents scoring
   this way had this label". See :func:`fit_calibrator`.
2. **Turning a calibrated probability into words** (request time). The UI never
   shows "87.32% AI". It shows a class and a confidence band, because a
   two-decimal percentage implies a precision this system does not have.

Banding rules and the abstain rule are here rather than scattered through the
detector, so that "when do we say Insufficient Evidence?" has exactly one answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

CONFIDENCE_BANDS: tuple[tuple[float, str], ...] = (
    (0.85, "high"),
    (0.65, "moderate"),
    (0.45, "low"),
)

# Below this top-class probability we decline to name a class at all.
ABSTAIN_MAX_PROBABILITY = 0.45
# Or when the top two classes are this close, the ranking is not meaningful.
ABSTAIN_MARGIN = 0.10
# Short documents do not carry enough measurement to support a verdict. 5
# sentences / 120 words is already generous for a ~250-word essay.
MIN_SENTENCES_FOR_VERDICT = 5
MIN_WORDS_FOR_VERDICT = 120

CLASS_LABELS: dict[str, str] = {
    "human": "Likely human-written",
    "ai_generated": "Likely AI-generated",
    "ai_polished": "Potentially AI-polished",
    "insufficient_evidence": "Insufficient evidence",
}

CLASS_DESCRIPTIONS: dict[str, str] = {
    "human": (
        "The measured writing patterns are consistent with the human-written "
        "examples in our evaluation data."
    ),
    "ai_generated": (
        "This text contains statistical patterns associated with the fully "
        "machine-generated examples in our evaluation data."
    ),
    "ai_polished": (
        "The patterns are most consistent with human writing that has been "
        "edited or rewritten with machine assistance."
    ),
    "insufficient_evidence": (
        "The measurements do not distinguish between the possibilities clearly "
        "enough to support any conclusion about this text."
    ),
}


@dataclass(slots=True)
class Verdict:
    classification: str
    label: str
    description: str
    confidence: str
    confidence_score: float
    probabilities: dict[str, float]
    margin: float
    abstained: bool
    abstain_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "confidence_score": round(self.confidence_score, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "margin": round(self.margin, 4),
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
        }


def confidence_band(probability: float) -> str:
    for threshold, name in CONFIDENCE_BANDS:
        if probability >= threshold:
            return name
    return "very low"


def build_verdict(
    probabilities: dict[str, float],
    *,
    n_sentences: int,
    n_words: int,
) -> Verdict:
    """Turn calibrated class probabilities into a user-facing verdict.

    Abstains — reporting ``insufficient_evidence`` — when the document is too
    short to measure, when no class is clearly ahead, or when the top two classes
    are within :data:`ABSTAIN_MARGIN`. Abstaining is a feature: a detector that
    always names a class is overstating what it knows.
    """
    if not probabilities:
        return Verdict(
            classification="insufficient_evidence",
            label=CLASS_LABELS["insufficient_evidence"],
            description=CLASS_DESCRIPTIONS["insufficient_evidence"],
            confidence="very low",
            confidence_score=0.0,
            probabilities={},
            margin=0.0,
            abstained=True,
            abstain_reason="No classifier output was available.",
        )

    ordered = sorted(probabilities.items(), key=lambda kv: -kv[1])
    top_class, top_probability = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = top_probability - runner_up

    reason: str | None = None
    if n_sentences < MIN_SENTENCES_FOR_VERDICT or n_words < MIN_WORDS_FOR_VERDICT:
        reason = (
            f"The text has {n_words} words in {n_sentences} sentences. "
            f"At least {MIN_WORDS_FOR_VERDICT} words and "
            f"{MIN_SENTENCES_FOR_VERDICT} sentences are needed before the "
            "distributional measurements are stable."
        )
    elif top_probability < ABSTAIN_MAX_PROBABILITY:
        reason = (
            f"The strongest class reached only {top_probability:.0%}, below the "
            f"{ABSTAIN_MAX_PROBABILITY:.0%} threshold required to name a class."
        )
    elif margin < ABSTAIN_MARGIN:
        reason = (
            f"The two leading possibilities are within {margin:.0%} of each other, "
            "so the ranking between them is not meaningful."
        )

    if reason:
        return Verdict(
            classification="insufficient_evidence",
            label=CLASS_LABELS["insufficient_evidence"],
            description=CLASS_DESCRIPTIONS["insufficient_evidence"],
            confidence="very low",
            confidence_score=float(top_probability),
            probabilities=probabilities,
            margin=float(margin),
            abstained=True,
            abstain_reason=reason,
        )

    return Verdict(
        classification=top_class,
        label=CLASS_LABELS.get(top_class, top_class),
        description=CLASS_DESCRIPTIONS.get(top_class, ""),
        confidence=confidence_band(top_probability),
        confidence_score=float(top_probability),
        probabilities=probabilities,
        margin=float(margin),
        abstained=False,
    )


# --------------------------------------------------------------------------- #
# Sentence-level banding
# --------------------------------------------------------------------------- #
SENTENCE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.75, "likely_ai_assisted", "high"),
    (0.60, "possibly_ai_assisted", "moderate"),
    (0.40, "uncertain", "low"),
    (0.25, "likely_human", "moderate"),
    (0.0, "likely_human", "high"),
)


def sentence_band(score: float, *, n_words: int, n_lm_tokens: float) -> tuple[str, str]:
    """``(classification, confidence)`` for one sentence.

    Very short sentences are forced to ``uncertain``: a four-word sentence gives
    the language model three predictions to work from, which is not enough to
    support a claim either way. This is the single most common source of
    misleading sentence-level highlighting in detectors that skip the check.
    """
    if n_words < 5 or n_lm_tokens < 5:
        return "uncertain", "very low"
    for threshold, classification, confidence in SENTENCE_BANDS:
        if score >= threshold:
            return classification, confidence
    return "uncertain", "low"


# --------------------------------------------------------------------------- #
# Calibration fitting (used by the training pipeline)
# --------------------------------------------------------------------------- #
def fit_calibrator(
    estimator: Any,
    X_calibration: np.ndarray,
    y_calibration: np.ndarray,
    *,
    method: str = "sigmoid",
) -> tuple[Any, str]:
    """Wrap a fitted estimator in a calibrator fit on held-out data.

    ``sigmoid`` (Platt scaling) is the default because the calibration split here
    has on the order of 70 documents; isotonic regression is non-parametric and
    would happily memorise a set that small. Isotonic is available via ``method``
    for larger corpora and is the better choice once the calibration split has a
    few hundred examples per class.

    Returns ``(calibrated_or_original, method_used)`` — if calibration cannot be
    fit (a class missing from the split, for example) the uncalibrated estimator
    is returned with method ``"none"``. Any failure is logged with its cause: a
    silently uncalibrated model is worse than a loud one, because the UI would go
    on presenting raw scores as if they were calibrated probabilities.
    """
    from sklearn.calibration import CalibratedClassifierCV

    from app.core.logging import get_logger, log_event

    logger = get_logger("app.calibration")

    if method == "none":
        return estimator, "none"

    classes_present = np.unique(y_calibration)
    if len(classes_present) < 2 or len(y_calibration) < 20:
        log_event(
            logger,
            "calibration.skipped",
            level="warning",
            reason="too few calibration samples or classes",
            samples=int(len(y_calibration)),
            classes=int(len(classes_present)),
        )
        return estimator, "none"
    if method == "isotonic" and min(np.bincount(y_calibration)) < 10:
        # Isotonic regression is non-parametric and will memorise a tiny class.
        log_event(
            logger,
            "calibration.downgraded",
            level="warning",
            requested="isotonic",
            used="sigmoid",
            smallest_class=int(min(np.bincount(y_calibration))),
        )
        method = "sigmoid"

    try:
        calibrated = CalibratedClassifierCV(
            _freeze(estimator), method=method, cv=_PREFIT_CV
        )
        calibrated.fit(X_calibration, y_calibration)
        return calibrated, method
    except Exception as exc:
        log_event(
            logger,
            "calibration.failed",
            level="error",
            method=method,
            type=type(exc).__name__,
            detail=str(exc)[:200],
        )
        return estimator, "none"


# scikit-learn 1.6 introduced ``FrozenEstimator`` and 1.9 removed the old
# ``cv="prefit"`` spelling entirely. Support both so the artifact can be
# retrained on either version.
try:  # sklearn >= 1.6
    from sklearn.frozen import FrozenEstimator as _FrozenEstimator

    _PREFIT_CV = None

    def _freeze(estimator: Any) -> Any:
        return _FrozenEstimator(estimator)

except ImportError:  # pragma: no cover - sklearn < 1.6
    _PREFIT_CV = "prefit"

    def _freeze(estimator: Any) -> Any:
        return estimator


def expected_calibration_error(
    probabilities: np.ndarray, y_true: np.ndarray, *, n_bins: int = 10
) -> float:
    """ECE of the predicted top class over equal-width confidence bins."""
    if probabilities.size == 0:
        return 0.0
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        error += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def reliability_curve(
    probabilities: np.ndarray, y_true: np.ndarray, *, n_bins: int = 10
) -> list[dict[str, float]]:
    """Points for the calibration plot in the research dashboard."""
    if probabilities.size == 0:
        return []
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = (predicted == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    points: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        points.append(
            {
                "bin_lower": round(float(lo), 3),
                "bin_upper": round(float(hi), 3),
                "mean_confidence": round(float(confidence[mask].mean()), 4),
                "observed_accuracy": round(float(correct[mask].mean()), 4),
                "count": int(mask.sum()),
            }
        )
    return points
