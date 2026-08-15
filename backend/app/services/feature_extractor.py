"""Feature assembly: raw text in, two fixed-width feature vectors out.

This module is the single source of truth for the feature space. Both the
offline training pipeline (``ml/training/extract_features.py``) and the live API
call it, which is what guarantees train/serve consistency — there is no second
implementation to drift.

Two vectors are produced per essay:

**Sentence vector** (``sentence_feature_names()``)
    ``lm_ + sty_ + syn_ + ctx_ + cor_``. Used by the sentence-level scorer for
    highlighting. Includes the ``ctx_*`` deviation features, so a sentence is
    always judged partly relative to its own document.

**Document vector** (``document_feature_names()``)
    Aggregates of the sentence blocks (mean/std/min/max/p25/p75 for a curated
    subset) plus document-only blocks ``bur_ + rep_ + shift_ + doc_ + cor_``.
    Used by the primary three-class classifier.

Feature groups are declared in :data:`FEATURE_GROUPS` so the ablation study
(stylometry-only vs LM-only vs hybrid) selects columns by group rather than by
retraining a differently-shaped pipeline.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger, log_event
from app.services import burstiness, repetition, style_shift
from app.services.corpus_analyzer import (
    COR_FEATURE_NAMES,
    CorpusReference,
    aggregate_document_views,
)
from app.services.document import ParsedDocument
from app.services.nlp import nlp_pipeline, segment
from app.services.probability_analyzer import (
    LM_FEATURE_NAMES,
    TokenScoreSet,
    aggregate_tokens,
    assign_to_sentences,
    lm_service,
)
from app.services.style_shift import (
    CTX_FEATURE_NAMES,
    SHIFT_FEATURE_NAMES,
)
from app.services.stylometry import (
    STY_FEATURE_NAMES,
    SYN_FEATURE_NAMES,
    measure_sentence,
)

logger = get_logger("app.features")

FEATURES_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Document-level structural features
# --------------------------------------------------------------------------- #
DOC_FEATURE_NAMES: tuple[str, ...] = (
    "doc_log_n_sentences",
    "doc_log_n_words",
    "doc_log_n_chars",
    "doc_n_paragraphs",
    "doc_mean_paragraph_words",
    "doc_std_paragraph_words",
    "doc_cv_paragraph_words",
    "doc_mean_sentences_per_paragraph",
    "doc_cv_sentences_per_paragraph",
    "doc_max_paragraph_words",
    "doc_min_paragraph_words",
    "doc_words_per_sentence",
    "doc_type_token_ratio",
    "doc_root_type_token_ratio",
    "doc_hapax_ratio",
    "doc_mean_syllables_per_word",
    "doc_flesch_reading_ease",
    "doc_flesch_kincaid_grade",
    "doc_paragraph_first_sentence_len_cv",
    "doc_has_single_paragraph",
)

# Sentence blocks aggregated into the document vector. The list is curated: the
# full cross-product of ~200 sentence features x 6 statistics would be ~1,200
# columns, which over-parameterises a few-hundred-document training set.
_AGG_STATS: tuple[str, ...] = ("mean", "std", "min", "max", "p25", "p75")
_AGG_FULL: tuple[str, ...] = (
    "lm_mean_logprob",
    "lm_median_logprob",
    "lm_std_logprob",
    "lm_log_perplexity",
    "lm_mean_entropy",
    "lm_frac_top1",
    "lm_frac_top10",
    "lm_frac_prob_gt_50",
    "lm_frac_prob_lt_1",
    "lm_mean_log_rank",
    "lm_mean_top1_gap",
    "lm_mean_normalised_surprisal",
    "sty_n_words",
    "sty_mean_word_len",
    "sty_root_ttr",
    "sty_stopword_ratio",
    "sty_punct_density",
    "sty_comma_rate",
    "syn_mean_dep_depth",
    "syn_n_clauses",
)
_AGG_MEAN_STD: tuple[str, ...] = (
    "lm_std_entropy",
    "lm_logprob_iqr",
    "lm_frac_top100",
    "lm_frac_rank_gt_1000",
    "lm_prob_variance",
    "lm_min_logprob",
    "lm_p10_logprob",
    "sty_ttr",
    "sty_hapax_ratio",
    "sty_long_word_ratio",
    "sty_short_word_ratio",
    "sty_syllables_per_word",
    "sty_std_word_len",
    "sty_content_word_ratio",
    "sty_semicolon_rate",
    "sty_colon_rate",
    "sty_dash_rate",
    "sty_paren_rate",
    "sty_quote_rate",
    "sty_exclaim_rate",
    "sty_question_rate",
    "sty_contraction_rate",
    "sty_colloquial_rate",
    "sty_llm_phrase_rate",
    "sty_transition_word_rate",
    "sty_transition_phrase_rate",
    "sty_sentence_initial_transition",
    "sty_hedge_rate",
    "sty_intensifier_rate",
    "sty_first_person_rate",
    "sty_nominalization_rate",
    "sty_avg_commas_per_clause",
    "sty_capitalised_word_ratio",
    "syn_max_dep_depth",
    "syn_mean_dep_distance",
    "syn_passive_ratio",
    "syn_subordinate_ratio",
    "syn_coordination_ratio",
    "syn_noun_verb_ratio",
    "syn_adj_noun_ratio",
    "syn_adverb_verb_ratio",
    "syn_function_word_ratio",
    "syn_pos_entropy",
    "syn_dep_entropy",
    "syn_starts_with_pronoun",
    "syn_starts_with_conj",
    "syn_starts_with_adverb",
    "syn_pos_adj",
    "syn_pos_adv",
    "syn_pos_aux",
    "syn_pos_cconj",
    "syn_pos_det",
    "syn_pos_noun",
    "syn_pos_pron",
    "syn_pos_propn",
    "syn_pos_sconj",
    "syn_pos_verb",
    "syn_pos_adp",
    "syn_pos_punct",
    "syn_dep_advcl",
    "syn_dep_advmod",
    "syn_dep_amod",
    "syn_dep_ccomp",
    "syn_dep_compound",
    "syn_dep_conj",
    "syn_dep_relcl",
    "syn_dep_xcomp",
    "syn_dep_nsubj",
    "syn_dep_prep",
    "ctx_pos_js_to_doc",
    "ctx_style_distance_to_doc",
    "ctx_funcword_cosine_to_doc",
    "ctx_abs_len_diff_prev",
)


def _aggregate_names() -> tuple[str, ...]:
    names: list[str] = []
    for feature in _AGG_FULL:
        names.extend(f"agg_{stat}_{feature}" for stat in _AGG_STATS)
    for feature in _AGG_MEAN_STD:
        names.extend((f"agg_mean_{feature}", f"agg_std_{feature}"))
    return tuple(names)


AGG_FEATURE_NAMES: tuple[str, ...] = _aggregate_names()

# Whole-document LM/stylometry measured on the document as one unit (not an
# aggregate of sentences) — captures long-range predictability.
WHOLE_DOC_LM_NAMES: tuple[str, ...] = tuple(f"whole_{n}" for n in LM_FEATURE_NAMES)


def sentence_feature_names() -> tuple[str, ...]:
    return (
        LM_FEATURE_NAMES
        + STY_FEATURE_NAMES
        + SYN_FEATURE_NAMES
        + CTX_FEATURE_NAMES
        + COR_FEATURE_NAMES
    )


def document_feature_names() -> tuple[str, ...]:
    return (
        AGG_FEATURE_NAMES
        + WHOLE_DOC_LM_NAMES
        + burstiness.BUR_FEATURE_NAMES
        + repetition.REP_FEATURE_NAMES
        + SHIFT_FEATURE_NAMES
        + DOC_FEATURE_NAMES
        + COR_FEATURE_NAMES
    )


# --------------------------------------------------------------------------- #
# Feature groups for the ablation study
# --------------------------------------------------------------------------- #
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Prefixes are matched against the *underlying* feature name, so an
    # aggregate like `agg_mean_lm_mean_logprob` belongs to the `lm` group.
    "lm": ("lm_",),
    "stylometric": ("sty_",),
    "syntactic": ("syn_",),
    "burstiness": ("bur_",),
    "repetition": ("rep_",),
    "structural": ("doc_",),
    "style_shift": ("ctx_", "shift_"),
    "corpus": ("cor_",),
}

MODEL_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    # Baseline: surface writing statistics only. No language model at all.
    "baseline_stylometric": (
        "stylometric",
        "syntactic",
        "burstiness",
        "repetition",
        "structural",
    ),
    # Isolates the contribution of the language-model instrument.
    "lm_only": ("lm",),
    # Everything.
    "hybrid": (
        "lm",
        "stylometric",
        "syntactic",
        "burstiness",
        "repetition",
        "structural",
        "style_shift",
        "corpus",
    ),
    # Hybrid minus within-document style shift, to answer research question 4.
    "hybrid_no_shift": (
        "lm",
        "stylometric",
        "syntactic",
        "burstiness",
        "repetition",
        "structural",
        "corpus",
    ),
}


def _base_name(feature: str) -> str:
    """Strip aggregation wrappers so group matching sees the underlying block."""
    for prefix in ("agg_mean_", "agg_std_", "agg_min_", "agg_max_", "agg_p25_", "agg_p75_"):
        if feature.startswith(prefix):
            return feature[len(prefix) :]
    if feature.startswith("whole_"):
        return feature[len("whole_") :]
    return feature


def group_of(feature: str) -> str | None:
    base = _base_name(feature)
    for group, prefixes in FEATURE_GROUPS.items():
        if base.startswith(prefixes):
            return group
    return None


def select_columns(feature_names: list[str] | tuple[str, ...], groups: tuple[str, ...]) -> list[int]:
    """Indices of the columns belonging to ``groups``."""
    wanted = set(groups)
    return [i for i, name in enumerate(feature_names) if group_of(name) in wanted]


# --------------------------------------------------------------------------- #
# Extraction result
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ExtractionResult:
    document: ParsedDocument
    sentence_features: list[dict[str, float]]
    document_features: dict[str, float]
    token_scores: TokenScoreSet
    rhythm: list[dict[str, float]] = field(default_factory=list)
    repeated_phrases: list[dict[str, Any]] = field(default_factory=list)
    repeated_templates: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    backend: str = "spacy"

    @property
    def n_sentences(self) -> int:
        return len(self.document.sentences)


class FeatureExtractor:
    """Runs the whole measurement pipeline over one essay."""

    def __init__(self, corpus_reference: CorpusReference | None = None) -> None:
        self.corpus_reference = corpus_reference

    def extract(self, text: str, *, with_corpus: bool = True) -> ExtractionResult:
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        document = segment(text)
        timings["segmentation_ms"] = _ms(t0)

        if not document.sentences:
            return ExtractionResult(
                document=document,
                sentence_features=[],
                document_features=dict.fromkeys(document_feature_names(), 0.0),
                token_scores=TokenScoreSet(),
                timings=timings,
                backend=document.segmentation_backend,
            )

        t0 = time.perf_counter()
        for sentence in document.sentences:
            measure_sentence(sentence)
        timings["stylometry_ms"] = _ms(t0)

        t0 = time.perf_counter()
        token_scores = lm_service.score_text(text)
        assign_to_sentences(document, token_scores)
        timings["lm_scoring_ms"] = _ms(t0)

        t0 = time.perf_counter()
        shift_features = style_shift.annotate(document)
        timings["style_shift_ms"] = _ms(t0)

        t0 = time.perf_counter()
        bur_features = burstiness.extract(document)
        rep_features = repetition.extract(document)
        rhythm = burstiness.sentence_rhythm(document)
        repeated_phrases = repetition.repeated_spans(document)
        repeated_templates = repetition.repeated_pos_templates(document)
        timings["rhythm_repetition_ms"] = _ms(t0)

        t0 = time.perf_counter()
        doc_views = aggregate_document_views(document)
        corpus_doc = dict.fromkeys(COR_FEATURE_NAMES, 0.0)
        reference = self.corpus_reference if with_corpus else None
        if reference is not None and reference.fitted:
            corpus_doc = reference.score(
                text,
                pos_sequence=doc_views["pos_sequence"],
                function_word_profile=doc_views["function_word_profile"],
                pos_distribution=doc_views["pos_distribution"],
            )
            for sentence in document.sentences:
                sentence.corpus = reference.score(
                    sentence.text,
                    pos_sequence=sentence.pos_sequence,
                    function_word_profile=sentence.function_word_profile,
                    pos_distribution=sentence.pos_distribution,
                )
        else:
            for sentence in document.sentences:
                sentence.corpus = dict.fromkeys(COR_FEATURE_NAMES, 0.0)
        timings["corpus_ms"] = _ms(t0)

        sentence_features = [
            {name: float(sentence.features().get(name, 0.0)) for name in sentence_feature_names()}
            for sentence in document.sentences
        ]

        whole_lm = {
            f"whole_{k}": v for k, v in aggregate_tokens(token_scores.tokens).items()
        }
        document_features = {
            **_aggregate_sentences(sentence_features),
            **whole_lm,
            **bur_features,
            **rep_features,
            **shift_features,
            **_structural_features(document),
            **corpus_doc,
        }
        document_features = {
            name: float(document_features.get(name, 0.0)) for name in document_feature_names()
        }

        timings["total_features_ms"] = round(sum(timings.values()), 2)
        log_event(
            logger,
            "features.extracted",
            sentences=len(document.sentences),
            paragraphs=len(document.paragraphs),
            sentence_features=len(sentence_feature_names()),
            document_features=len(document_features),
            backend=document.segmentation_backend,
            duration_ms=timings["total_features_ms"],
        )

        return ExtractionResult(
            document=document,
            sentence_features=sentence_features,
            document_features=document_features,
            token_scores=token_scores,
            rhythm=rhythm,
            repeated_phrases=repeated_phrases,
            repeated_templates=repeated_templates,
            timings=timings,
            backend=document.segmentation_backend,
        )

    def warmup(self) -> dict[str, Any]:
        """Force model loading so the first real request is not slow."""
        started = time.perf_counter()
        nlp_pipeline.load()
        lm_service.load()
        # A short pass through the whole pipeline compiles the lazy code paths.
        self.extract(
            "I built a small robot in my garage. It never worked properly, but I "
            "learned to solder. That summer changed how I think about failure.",
            with_corpus=False,
        )
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        log_event(logger, "pipeline.warmed_up", duration_ms=elapsed)
        return {"warmup_ms": elapsed, "lm": lm_service.info(), "spacy": nlp_pipeline.info()}


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return dict.fromkeys(_AGG_STATS, 0.0)
    n = len(values)
    mu = sum(values) / n
    sigma = math.sqrt(sum((v - mu) ** 2 for v in values) / n) if n > 1 else 0.0
    ordered = sorted(values)

    def pct(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    return {
        "mean": mu,
        "std": sigma,
        "min": ordered[0],
        "max": ordered[-1],
        "p25": pct(0.25),
        "p75": pct(0.75),
    }


def _aggregate_sentences(sentence_features: list[dict[str, float]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for feature in _AGG_FULL:
        stats = _stats([sf.get(feature, 0.0) for sf in sentence_features])
        for stat in _AGG_STATS:
            out[f"agg_{stat}_{feature}"] = stats[stat]
    for feature in _AGG_MEAN_STD:
        stats = _stats([sf.get(feature, 0.0) for sf in sentence_features])
        out[f"agg_mean_{feature}"] = stats["mean"]
        out[f"agg_std_{feature}"] = stats["std"]
    return out


def _structural_features(document: ParsedDocument) -> dict[str, float]:
    from app.services.stylometry import count_syllables, words_of

    words = words_of(document.text)
    n_words = len(words) or 1
    n_sentences = max(len(document.sentences), 1)
    unique = len({w.lower() for w in words})
    hapax = len([w for w, c in _counter(words).items() if c == 1])
    syllables = sum(count_syllables(w) for w in words)
    spw = syllables / n_words
    wps = n_words / n_sentences

    paragraph_words = [
        float(sum(document.sentences[i].n_words for i in p.sentence_indices))
        for p in document.paragraphs
    ] or [float(n_words)]
    sentences_per_paragraph = [float(p.n_sentences) for p in document.paragraphs] or [
        float(n_sentences)
    ]
    first_sentence_lengths = [
        float(document.sentences[p.sentence_indices[0]].n_words)
        for p in document.paragraphs
        if p.sentence_indices
    ] or [wps]

    mean_para = sum(paragraph_words) / len(paragraph_words)
    std_para = _std_of(paragraph_words)
    mean_spp = sum(sentences_per_paragraph) / len(sentences_per_paragraph)

    return {
        "doc_log_n_sentences": math.log1p(len(document.sentences)),
        "doc_log_n_words": math.log1p(n_words),
        "doc_log_n_chars": math.log1p(len(document.text)),
        "doc_n_paragraphs": float(len(document.paragraphs)),
        "doc_mean_paragraph_words": mean_para,
        "doc_std_paragraph_words": std_para,
        "doc_cv_paragraph_words": std_para / mean_para if mean_para > 1e-9 else 0.0,
        "doc_mean_sentences_per_paragraph": mean_spp,
        "doc_cv_sentences_per_paragraph": (
            _std_of(sentences_per_paragraph) / mean_spp if mean_spp > 1e-9 else 0.0
        ),
        "doc_max_paragraph_words": max(paragraph_words),
        "doc_min_paragraph_words": min(paragraph_words),
        "doc_words_per_sentence": wps,
        "doc_type_token_ratio": unique / n_words,
        "doc_root_type_token_ratio": unique / math.sqrt(n_words),
        "doc_hapax_ratio": hapax / n_words,
        "doc_mean_syllables_per_word": spw,
        # Flesch formulas: standard, transparent readability measures. They are
        # included because "polish this" edits reliably move them.
        "doc_flesch_reading_ease": 206.835 - 1.015 * wps - 84.6 * spw,
        "doc_flesch_kincaid_grade": 0.39 * wps + 11.8 * spw - 15.59,
        "doc_paragraph_first_sentence_len_cv": (
            _std_of(first_sentence_lengths)
            / (sum(first_sentence_lengths) / len(first_sentence_lengths))
            if first_sentence_lengths and sum(first_sentence_lengths) > 0
            else 0.0
        ),
        "doc_has_single_paragraph": 1.0 if len(document.paragraphs) <= 1 else 0.0,
    }


def _counter(words: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for word in words:
        key = word.lower()
        counts[key] = counts.get(key, 0) + 1
    return counts


def _std_of(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))
