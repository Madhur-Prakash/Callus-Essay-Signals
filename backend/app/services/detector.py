"""Detector orchestration: essay text in, explained verdict out.

The full path, in order:

    normalise -> segment -> stylometry -> LM token scoring -> style shift
      -> burstiness/repetition -> corpus similarity -> feature vectors
      -> our classifier -> calibration -> banding -> evidence -> result

The classification is produced in :mod:`app.services.classifier` from our own
trained scikit-learn model. The language model contributes token probabilities
and nothing else; there is no code path from this module to a hosted chat model.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.config import Settings, get_settings
from app.core.logging import content_hash, get_logger, log_event, safe_text_meta
from app.services.calibration import build_verdict, sentence_band
from app.services.classifier import detector_models
from app.services.corpus_analyzer import CorpusReference
from app.services.explanation_engine import (
    EXPLANATION_ENGINE_VERSION,
    ExplanationEngine,
    essay_context_from,
)
from app.services.feature_extractor import FEATURES_VERSION, FeatureExtractor
from app.services.nlp import nlp_pipeline
from app.services.probability_analyzer import lm_service, token_evidence

logger = get_logger("app.detector")

# A sentence at or above this machine-likeness counts as "flagged" in the
# summary counts. It matches the `possibly_ai_assisted` band boundary in
# app.services.calibration.SENTENCE_BANDS.
FLAG_THRESHOLD = 0.60
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.40, 0.60


@dataclass(slots=True)
class AnalysisResult:
    analysis_id: str
    status: str
    verdict: dict[str, Any]
    summary: dict[str, Any]
    paragraphs: list[dict[str, Any]]
    sentences: list[dict[str, Any]]
    evidence: dict[str, Any]
    rhythm: list[dict[str, float]]
    repetition: dict[str, Any]
    model_info: dict[str, Any]
    timings: dict[str, float]
    content_hash: str
    created_at: str
    persisted: bool = False
    cached: bool = False
    warnings: list[str] = field(default_factory=list)
    text: str | None = None
    """Only populated when ``SAVE_ESSAYS=true``; never logged either way."""

    def to_dict(self, *, include_text: bool = False) -> dict[str, Any]:
        payload = {
            "analysis_id": self.analysis_id,
            "status": self.status,
            **self.verdict,
            "summary": self.summary,
            "paragraphs": self.paragraphs,
            "sentences": self.sentences,
            "evidence": self.evidence,
            "rhythm": self.rhythm,
            "repetition": self.repetition,
            "model": self.model_info,
            "timings": self.timings,
            "content_hash": self.content_hash,
            "created_at": self.created_at,
            "persisted": self.persisted,
            "cached": self.cached,
            "warnings": self.warnings,
        }
        if include_text and self.text is not None:
            payload["text"] = self.text
        return payload


class Detector:
    """Owns the pipeline objects for the process lifetime."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.corpus_reference: CorpusReference | None = None
        self.extractor = FeatureExtractor(corpus_reference=None)
        self.explanation = ExplanationEngine()
        self._warmed = False

    # ------------------------------------------------------------- lifecycle
    def load(self) -> dict[str, Any]:
        """Load models and artifacts. Safe to call more than once."""
        artifacts = self.settings.artifacts_path
        self.corpus_reference = CorpusReference.load(artifacts)
        self.extractor.corpus_reference = self.corpus_reference
        detector_models.load(artifacts)
        self.explanation = ExplanationEngine(detector_models.reference_stats)
        return self.status()

    def warmup(self) -> dict[str, Any]:
        info = self.extractor.warmup()
        self._warmed = True
        return info

    def status(self) -> dict[str, Any]:
        return {
            "models_ready": detector_models.ready,
            "model_error": detector_models.load_error,
            "corpus_reference": (
                self.corpus_reference.summary()
                if self.corpus_reference is not None
                else {"fitted": False}
            ),
            "explanation_reference": self.explanation.has_reference,
            "language_model": lm_service.info(),
            "spacy": nlp_pipeline.info(),
            "warmed_up": self._warmed,
            "versions": {
                "detector": self.settings.detector_version,
                "features": FEATURES_VERSION,
                "explanation_engine": EXPLANATION_ENGINE_VERSION,
            },
        }

    # -------------------------------------------------------------- analysis
    def analyse(self, text: str, *, analysis_id: str | None = None) -> AnalysisResult:
        """Run the full pipeline. ``text`` must already be validated."""
        settings = self.settings
        analysis_id = analysis_id or uuid.uuid4().hex
        started = time.perf_counter()

        model_meta = detector_models.metadata
        digest = content_hash(
            text,
            detector_version=settings.detector_version,
            model_version=str(model_meta.get("model_version", "untrained")),
        )
        log_event(
            logger,
            "analysis.start",
            analysis_id=analysis_id,
            **safe_text_meta(text, prefix="essay"),
        )

        extraction = self.extractor.extract(text)
        document = extraction.document
        warnings: list[str] = []
        if extraction.backend != "spacy":
            warnings.append(
                "The spaCy model was unavailable, so part-of-speech and dependency "
                "features could not be measured. Results are degraded."
            )
        if self.corpus_reference is None or not self.corpus_reference.fitted:
            warnings.append(
                "No reference corpus is loaded, so cross-corpus similarity features "
                "were not measured."
            )
        regime = model_meta.get("data_regime")
        if regime == "bootstrap":
            warnings.append(
                "This detector is trained on the offline bootstrap corpus. Its "
                "reported accuracy reflects separability of the bootstrap "
                "generators, not real-world performance. Treat any verdict as a "
                "demonstration, not a measurement."
            )

        # ----------------------------------------------------- classification
        classify_started = time.perf_counter()
        probabilities = detector_models.predict_document(extraction.document_features)
        sentence_scores = detector_models.predict_sentences(extraction.sentence_features)
        classify_ms = round((time.perf_counter() - classify_started) * 1000, 2)

        if not sentence_scores:
            warnings.append(
                "The sentence-level model is unavailable, so per-sentence "
                "highlighting is not shown."
            )
            sentence_scores = []

        verdict = build_verdict(
            probabilities,
            n_sentences=document.n_sentences,
            n_words=document.n_words,
        )

        # -------------------------------------------------------- sentences
        context = essay_context_from(document, extraction.rhythm)
        token_buckets = _tokens_by_sentence(document, extraction)
        sentences: list[dict[str, Any]] = []

        for i, sentence in enumerate(document.sentences):
            score = float(sentence_scores[i]) if i < len(sentence_scores) else None
            features = extraction.sentence_features[i]
            n_lm_tokens = float(features.get("lm_n_tokens", 0.0))
            if score is None:
                classification, confidence = "unavailable", "none"
            else:
                classification, confidence = sentence_band(
                    score, n_words=sentence.n_words, n_lm_tokens=n_lm_tokens
                )

            entry: dict[str, Any] = {
                "sentence_id": sentence.index,
                "paragraph_id": sentence.paragraph_index,
                "start": sentence.start,
                "end": sentence.end,
                "text": sentence.text,
                "score": round(score, 4) if score is not None else None,
                "classification": classification,
                "confidence": confidence,
                "n_words": sentence.n_words,
                "features": _sentence_feature_digest(features),
            }
            # Evidence is only attached where it will be read - flagged and
            # uncertain sentences. Attaching it to every sentence would triple the
            # response size for content the UI never shows.
            if score is not None and score >= UNCERTAIN_LOW:
                evidence = self.explanation.explain_sentence(
                    features,
                    essay_context=context,
                    score=score,
                    contributions=detector_models.sentence_contributions(features),
                    token_evidence=token_evidence(token_buckets.get(sentence.index, [])),
                )
                entry["evidence"] = evidence.to_dict()
            sentences.append(entry)

        paragraphs = _paragraph_rollup(document, sentences)
        flagged_paragraph_ids = [
            p["paragraph_id"] for p in paragraphs if p["classification"] != "likely_human"
        ]

        # ---------------------------------------------------- document evidence
        doc_evidence = self.explanation.explain_document(
            extraction.document_features,
            rhythm=extraction.rhythm,
            repeated_phrases=extraction.repeated_phrases,
            repeated_templates=extraction.repeated_templates,
            contributions=detector_models.document_contributions(
                extraction.document_features
            ),
            flagged_paragraphs=flagged_paragraph_ids,
        )

        summary = _build_summary(document, sentences, paragraphs, extraction)
        total_ms = round((time.perf_counter() - started) * 1000, 2)
        timings = {
            **extraction.timings,
            "classification_ms": classify_ms,
            "total_ms": total_ms,
        }

        result = AnalysisResult(
            analysis_id=analysis_id,
            status="completed",
            verdict=verdict.to_dict(),
            summary=summary,
            paragraphs=paragraphs,
            sentences=sentences,
            evidence=doc_evidence.to_dict(),
            rhythm=extraction.rhythm,
            repetition={
                "repeated_phrases": extraction.repeated_phrases,
                "repeated_syntactic_templates": extraction.repeated_templates,
            },
            model_info=_model_info(self.settings, model_meta),
            timings=timings if self.settings.debug_timings else {"total_ms": total_ms},
            content_hash=digest,
            created_at=datetime.now(UTC).isoformat(),
            warnings=warnings,
            text=text if self.settings.save_essays else None,
        )

        log_event(
            logger,
            "analysis.complete",
            analysis_id=analysis_id,
            classification=verdict.classification,
            confidence=verdict.confidence,
            abstained=verdict.abstained,
            sentences=document.n_sentences,
            paragraphs=document.n_paragraphs,
            flagged_sentences=summary["flagged_sentences"],
            duration_ms=total_ms,
        )
        return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _tokens_by_sentence(document, extraction) -> dict[int, list]:  # noqa: ANN001
    """Group scored LM tokens by the sentence that contains them."""
    buckets: dict[int, list] = {}
    bounds = [(s.start, s.end, s.index) for s in document.sentences]
    cursor = 0
    for token in extraction.token_scores.tokens:
        anchor = token.start + (len(token.text) - len(token.text.lstrip()))
        while cursor < len(bounds) - 1 and anchor >= bounds[cursor][1]:
            cursor += 1
        buckets.setdefault(bounds[cursor][2], []).append(token)
    return buckets


def _sentence_feature_digest(features: dict[str, float]) -> dict[str, float]:
    """The subset of sentence features the UI actually displays."""
    keys = (
        "lm_mean_logprob",
        "lm_perplexity",
        "lm_frac_top1",
        "lm_mean_entropy",
        "lm_n_tokens",
        "sty_n_words",
        "sty_root_ttr",
        "sty_mean_word_len",
        "sty_punct_density",
        "sty_comma_rate",
        "sty_transition_word_rate",
        "sty_nominalization_rate",
        "syn_n_clauses",
        "syn_mean_dep_depth",
        "ctx_z_mean_logprob",
        "ctx_z_n_words",
        "ctx_style_distance_to_doc",
        "ctx_pos_js_to_doc",
        "cor_char_sim_delta_ai_human",
    )
    return {k: round(float(features.get(k, 0.0)), 4) for k in keys}


def _paragraph_rollup(document, sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate sentence scores into paragraph verdicts.

    A paragraph is scored by the mean of its sentence scores weighted by word
    count - an eight-word aside should not swing a paragraph the way a forty-word
    sentence does. The maximum is reported separately, because a single strongly
    flagged sentence inside an otherwise ordinary paragraph is exactly the
    localised-edit pattern worth surfacing.
    """
    output: list[dict[str, Any]] = []
    for paragraph in document.paragraphs:
        members = [sentences[i] for i in paragraph.sentence_indices if i < len(sentences)]
        scored = [s for s in members if s["score"] is not None]
        if scored:
            weights = [max(1, s["n_words"]) for s in scored]
            weighted = sum(s["score"] * w for s, w in zip(scored, weights, strict=False)) / sum(
                weights
            )
            peak = max(s["score"] for s in scored)
            human_likeness = 1.0 - weighted
        else:
            weighted = peak = human_likeness = None

        flagged = [s["sentence_id"] for s in scored if s["score"] >= FLAG_THRESHOLD]
        uncertain = [
            s["sentence_id"]
            for s in scored
            if UNCERTAIN_LOW <= s["score"] < UNCERTAIN_HIGH
        ]

        if weighted is None:
            classification = "unavailable"
        elif weighted >= FLAG_THRESHOLD:
            classification = "likely_ai_assisted"
        elif weighted >= UNCERTAIN_LOW:
            classification = "uncertain"
        elif peak is not None and peak >= 0.75 and len(scored) >= 3:
            # Paragraph average looks human but one sentence stands out sharply.
            classification = "contains_flagged_sentence"
        else:
            classification = "likely_human"

        output.append(
            {
                "paragraph_id": paragraph.index,
                "start": paragraph.start,
                "end": paragraph.end,
                "n_sentences": len(members),
                "n_words": sum(s["n_words"] for s in members),
                "score": round(weighted, 4) if weighted is not None else None,
                "max_sentence_score": round(peak, 4) if peak is not None else None,
                "human_likeness": round(human_likeness, 4)
                if human_likeness is not None
                else None,
                "classification": classification,
                "flagged_sentence_ids": flagged,
                "uncertain_sentence_ids": uncertain,
                "sentence_ids": [s["sentence_id"] for s in members],
            }
        )
    return output


def _build_summary(document, sentences, paragraphs, extraction) -> dict[str, Any]:  # noqa: ANN001
    scored = [s for s in sentences if s["score"] is not None]
    flagged = [s for s in scored if s["score"] >= FLAG_THRESHOLD]
    uncertain = [s for s in scored if UNCERTAIN_LOW <= s["score"] < UNCERTAIN_HIGH]
    human_like = [s for s in scored if s["score"] < UNCERTAIN_LOW]
    features = extraction.document_features

    return {
        "n_words": document.n_words,
        "n_characters": len(document.text),
        "n_sentences": document.n_sentences,
        "n_paragraphs": document.n_paragraphs,
        "sentences_scored": len(scored),
        "flagged_sentences": len(flagged),
        "uncertain_sentences": len(uncertain),
        "human_like_sentences": len(human_like),
        "flagged_paragraphs": sum(
            1 for p in paragraphs if p["classification"] in {"likely_ai_assisted", "contains_flagged_sentence"}
        ),
        "uncertain_paragraphs": sum(1 for p in paragraphs if p["classification"] == "uncertain"),
        "flagged_share": round(len(flagged) / len(scored), 4) if scored else 0.0,
        "statistics": {
            "mean_words_per_sentence": round(features.get("doc_words_per_sentence", 0.0), 2),
            "sentence_length_std": round(features.get("bur_std_sent_len", 0.0), 2),
            "sentence_length_cv": round(features.get("bur_cv_sent_len", 0.0), 3),
            "burstiness_index": round(features.get("bur_burstiness_index", 0.0), 3),
            "perplexity": round(features.get("whole_lm_perplexity", 0.0), 2),
            "median_sentence_perplexity": round(
                features.get("agg_p75_lm_log_perplexity", 0.0), 3
            ),
            "fraction_top1_tokens": round(features.get("whole_lm_frac_top1", 0.0), 4),
            "mean_token_logprob": round(features.get("whole_lm_mean_logprob", 0.0), 4),
            "mean_token_entropy": round(features.get("whole_lm_mean_entropy", 0.0), 4),
            "type_token_ratio": round(features.get("doc_type_token_ratio", 0.0), 4),
            "root_type_token_ratio": round(features.get("doc_root_type_token_ratio", 0.0), 3),
            "trigram_repeat_ratio": round(features.get("rep_trigram_repeat_ratio", 0.0), 4),
            "pos_template_repeat_ratio": round(
                features.get("rep_pos_fourgram_repeat_ratio", 0.0), 4
            ),
            "max_style_shift": round(features.get("shift_max_style_distance_to_doc", 0.0), 3),
            "style_changepoints": int(features.get("shift_n_changepoints", 0.0)),
            "flesch_reading_ease": round(features.get("doc_flesch_reading_ease", 0.0), 1),
            "transition_word_rate": round(
                features.get("agg_mean_sty_transition_word_rate", 0.0), 3
            ),
            "contraction_rate": round(features.get("agg_mean_sty_contraction_rate", 0.0), 3),
        },
        "lm_tokens_scored": len(extraction.token_scores.tokens),
        "lm_windows": extraction.token_scores.n_windows,
        "segmentation_backend": extraction.backend,
    }


def _model_info(settings: Settings, metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "detector_version": settings.detector_version,
        "model_version": metadata.get("model_version", "untrained"),
        "dataset_version": metadata.get("dataset_version"),
        "features_version": FEATURES_VERSION,
        "explanation_engine_version": EXPLANATION_ENGINE_VERSION,
        "data_regime": metadata.get("data_regime"),
        "language_model": settings.lm_model_name,
        "language_model_role": (
            "instrument for token probabilities; it does not classify the essay"
        ),
        "classifier": (metadata.get("document_model") or {}).get("name"),
        "trained_at": metadata.get("trained_at"),
    }


detector = Detector()
