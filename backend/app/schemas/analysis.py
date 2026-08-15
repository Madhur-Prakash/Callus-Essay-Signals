"""Request and response models for the analysis API.

These types are the public contract: FastAPI derives the OpenAPI document from
them, and the frontend's TypeScript types mirror them field for field.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Classification = Literal[
    "human", "ai_generated", "ai_polished", "insufficient_evidence"
]
SentenceClassification = Literal[
    "likely_human",
    "uncertain",
    "possibly_ai_assisted",
    "likely_ai_assisted",
    "unavailable",
]
ParagraphClassification = Literal[
    "likely_human",
    "uncertain",
    "contains_flagged_sentence",
    "likely_ai_assisted",
    "unavailable",
]
Confidence = Literal["high", "moderate", "low", "very low", "none"]


class AnalysisRequest(BaseModel):
    """Analysis request.

    Note the absence of a blank/length check here. Bounds are owned by
    :func:`app.core.security.validate_essay`, which is configuration-driven and
    returns specific, actionable codes (``essay_empty``, ``essay_too_short``,
    ``essay_too_long``). Duplicating the check at the schema layer meant
    whitespace-only input was rejected with a generic ``validation_error``
    instead — two validators for one rule, and the less helpful one won.
    """

    text: str = Field(
        ...,
        description=(
            "The essay to analyse. Bounds come from MIN_ESSAY_CHARS / MAX_ESSAY_CHARS; "
            "empty or whitespace-only text returns `essay_empty`."
        ),
        examples=["I built a robot in my garage the summer before junior year..."],
    )
    save: bool | None = Field(
        default=None,
        description=(
            "Override the server's persistence default for this request. When false "
            "the essay text is never written to storage; only derived metrics are. "
            "The server's SAVE_ESSAYS setting is the ceiling — this flag can opt out, "
            "never in."
        ),
    )
    async_mode: bool = Field(
        default=False,
        description=(
            "Queue the analysis through Kafka instead of running it inline. Only "
            "honoured when Kafka is enabled; otherwise the request runs synchronously "
            "and the response says so."
        ),
    )


class Meter(BaseModel):
    key: str
    label: str
    strength: float = Field(ge=0.0, le=1.0)
    level: str
    value: float
    display: str = ""
    unit: str
    reference: str
    detail: str
    available: bool = True
    percentile_vs_human: float | None = None


class Measurement(BaseModel):
    name: str
    key: str
    value: float | int
    unit: str
    reference: str
    percentile_vs_human: float | None = None
    strength: float | None = None


class FeatureContribution(BaseModel):
    feature: str
    value: float
    standardised: float
    contribution: float
    direction: str
    method: str


class EvidenceBlock(BaseModel):
    meters: list[Meter] = Field(default_factory=list)
    statements: list[str] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    model_contributions: list[FeatureContribution] = Field(default_factory=list)
    engine_version: str = ""


class SentenceResult(BaseModel):
    sentence_id: int
    paragraph_id: int
    start: int = Field(description="Character offset into the normalised essay text.")
    end: int
    text: str
    score: float | None = Field(
        default=None,
        description=(
            "Calibrated machine-likeness in [0, 1] from the sentence-level model. "
            "Null when the sentence model is unavailable."
        ),
    )
    classification: SentenceClassification
    confidence: Confidence | str
    n_words: int
    features: dict[str, float] = Field(default_factory=dict)
    evidence: EvidenceBlock | None = Field(
        default=None,
        description="Present only for flagged and uncertain sentences.",
    )


class ParagraphResult(BaseModel):
    paragraph_id: int
    start: int
    end: int
    n_sentences: int
    n_words: int
    score: float | None = None
    max_sentence_score: float | None = None
    human_likeness: float | None = None
    classification: ParagraphClassification
    flagged_sentence_ids: list[int] = Field(default_factory=list)
    uncertain_sentence_ids: list[int] = Field(default_factory=list)
    sentence_ids: list[int] = Field(default_factory=list)


class RhythmPoint(BaseModel):
    index: int
    paragraph_index: int
    words: float
    deviation_from_mean: float
    abs_diff_prev: float
    clauses: float
    mean_logprob: float
    perplexity: float


class RepeatedPhrase(BaseModel):
    phrase: str
    length: int
    count: int
    sentence_indices: list[int] = Field(default_factory=list)


class RepeatedTemplate(BaseModel):
    template: str
    sentence_count: int
    sentence_indices: list[int] = Field(default_factory=list)


class RepetitionBlock(BaseModel):
    repeated_phrases: list[RepeatedPhrase] = Field(default_factory=list)
    repeated_syntactic_templates: list[RepeatedTemplate] = Field(default_factory=list)


class SummaryStatistics(BaseModel):
    mean_words_per_sentence: float
    sentence_length_std: float
    sentence_length_cv: float
    burstiness_index: float
    perplexity: float
    median_sentence_perplexity: float
    fraction_top1_tokens: float
    mean_token_logprob: float
    mean_token_entropy: float
    type_token_ratio: float
    root_type_token_ratio: float
    trigram_repeat_ratio: float
    pos_template_repeat_ratio: float
    max_style_shift: float
    style_changepoints: int
    flesch_reading_ease: float
    transition_word_rate: float
    contraction_rate: float


class AnalysisSummary(BaseModel):
    n_words: int
    n_characters: int
    n_sentences: int
    n_paragraphs: int
    sentences_scored: int
    flagged_sentences: int
    uncertain_sentences: int
    human_like_sentences: int
    flagged_paragraphs: int
    uncertain_paragraphs: int
    flagged_share: float
    statistics: SummaryStatistics
    lm_tokens_scored: int
    lm_windows: int
    segmentation_backend: str


class ModelBlock(BaseModel):
    detector_version: str
    model_version: str
    dataset_version: str | None = None
    features_version: str
    explanation_engine_version: str
    data_regime: str | None = None
    language_model: str
    language_model_role: str
    classifier: str | None = None
    trained_at: str | None = None


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: Literal["completed", "queued", "processing", "failed"]
    classification: Classification
    label: str
    description: str
    confidence: Confidence | str
    confidence_score: float
    probabilities: dict[str, float]
    margin: float
    abstained: bool
    abstain_reason: str | None = None
    summary: AnalysisSummary
    paragraphs: list[ParagraphResult]
    sentences: list[SentenceResult]
    evidence: EvidenceBlock
    rhythm: list[RhythmPoint]
    repetition: RepetitionBlock
    model: ModelBlock
    timings: dict[str, float] = Field(default_factory=dict)
    content_hash: str
    created_at: str
    persisted: bool
    cached: bool
    warnings: list[str] = Field(default_factory=list)
    disclaimer: str = Field(
        default=(
            "AI detection is probabilistic. A flag is evidence for human review, not "
            "proof of authorship. This system cannot establish who wrote a text and "
            "must not be used as the sole basis for any decision about a person."
        )
    )


class QueuedAnalysisResponse(BaseModel):
    analysis_id: str
    status: Literal["queued"]
    poll_url: str
    message: str
    content_hash: str
    created_at: str


class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: str
    created_at: str | None = None
    updated_at: str | None = None
    classification: str | None = None
    confidence: str | None = None
    error: str | None = None


class SentenceListResponse(BaseModel):
    analysis_id: str
    n_sentences: int
    sentences: list[SentenceResult]


class AnalysisListItem(BaseModel):
    analysis_id: str
    status: str
    classification: str | None = None
    confidence: str | None = None
    created_at: str | None = None
    n_words: int | None = None
    n_sentences: int | None = None
    flagged_sentences: int | None = None


class AnalysisListResponse(BaseModel):
    total: int
    items: list[AnalysisListItem]


class DeleteResponse(BaseModel):
    analysis_id: str
    deleted: bool


class ComponentHealth(BaseModel):
    name: str
    enabled: bool
    available: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    version: str
    environment: str
    components: list[ComponentHealth]
    detector: dict[str, Any]
    uptime_seconds: float
    checked_at: str


class ModelInfoResponse(BaseModel):
    ready: bool
    error: str | None = None
    detector_version: str
    model_version: str | None = None
    dataset_version: str | None = None
    features_version: str | None = None
    trained_at: str | None = None
    data_regime: str | None = None
    document_model: dict[str, Any]
    sentence_model: dict[str, Any]
    language_model: dict[str, Any]
    metrics: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    feature_importance: list[dict[str, Any]] = Field(default_factory=list)
    model_comparison: list[dict[str, Any]] = Field(default_factory=list)
    methodology: dict[str, Any] = Field(default_factory=dict)
    analysis_thresholds: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Hard input bounds and the soft thresholds below which the detector "
            "abstains. Clients should read these rather than hard-coding limits."
        ),
    )


class EvaluationResponse(BaseModel):
    available: bool
    report: dict[str, Any] | None = None
    failures: dict[str, Any] | None = None
    dataset: dict[str, Any] | None = None
    message: str | None = None
