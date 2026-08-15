"""Stylometric feature extraction: surface, lexical and syntactic.

Every function here is a pure measurement over text or a spaCy span. Nothing in
this module makes a judgement about authorship - it only produces numbers that
the trained classifier is free to weight (or ignore).

Feature name prefixes
---------------------
``sty_``  surface / lexical
``syn_``  POS, dependency and construction-level
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.services.lexicons import (
    CLAUSAL_DEPS,
    COLLOQUIAL_MARKERS,
    CONTRACTION_PATTERN,
    DEP_LABELS,
    FIRST_PERSON,
    FUNCTION_WORDS,
    HEDGES,
    INTENSIFIERS,
    LLM_REGISTER_PHRASES,
    NOMINALIZATION_SUFFIXES,
    PASSIVE_DEPS,
    POS_TAGS,
    STOPWORDS,
    TRANSITION_PHRASES,
    TRANSITION_WORDS,
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_CONTRACTION_RE = re.compile(CONTRACTION_PATTERN, re.IGNORECASE)
_TRANSITION_WORD_SET = frozenset(TRANSITION_WORDS)
_HEDGE_WORDS = frozenset(w for w in HEDGES if " " not in w)
_HEDGE_PHRASES = tuple(w for w in HEDGES if " " in w)
_INTENSIFIER_SET = frozenset(INTENSIFIERS)
_COLLOQUIAL_WORDS = frozenset(w for w in COLLOQUIAL_MARKERS if " " not in w)
_COLLOQUIAL_PHRASES = tuple(w for w in COLLOQUIAL_MARKERS if " " in w)
_FUNCTION_WORD_INDEX = {w: i for i, w in enumerate(FUNCTION_WORDS)}

STY_FEATURE_NAMES: tuple[str, ...] = (
    "sty_n_words",
    "sty_n_chars",
    "sty_mean_word_len",
    "sty_std_word_len",
    "sty_max_word_len",
    "sty_long_word_ratio",
    "sty_short_word_ratio",
    "sty_syllables_per_word",
    "sty_ttr",
    "sty_root_ttr",
    "sty_hapax_ratio",
    "sty_stopword_ratio",
    "sty_content_word_ratio",
    "sty_punct_density",
    "sty_comma_rate",
    "sty_semicolon_rate",
    "sty_colon_rate",
    "sty_dash_rate",
    "sty_paren_rate",
    "sty_quote_rate",
    "sty_exclaim_rate",
    "sty_question_rate",
    "sty_ellipsis_rate",
    "sty_digit_ratio",
    "sty_uppercase_ratio",
    "sty_capitalised_word_ratio",
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
)

SYN_FEATURE_NAMES: tuple[str, ...] = (
    tuple(f"syn_pos_{tag.lower()}" for tag in POS_TAGS)
    + tuple(f"syn_dep_{dep}" for dep in DEP_LABELS)
    + (
        "syn_n_clauses",
        "syn_clauses_per_sentence",
        "syn_max_dep_depth",
        "syn_mean_dep_depth",
        "syn_mean_dep_distance",
        "syn_max_dep_distance",
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
        "syn_starts_with_prep",
        "syn_ends_with_period",
    )
)


# --------------------------------------------------------------------------- #
# Small numeric helpers
# --------------------------------------------------------------------------- #
def _mean(values) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _std(values) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    mu = sum(values) / len(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _rate(count: int, total: int, per: float = 1.0) -> float:
    return float(count * per / total) if total else 0.0


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log2(p)
    return float(h)


def count_syllables(word: str) -> int:
    """Cheap English syllable heuristic (vowel groups, silent-e correction)."""
    w = word.lower().strip("'-")
    if not w:
        return 0
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    return max(1, n)


def words_of(text: str) -> list[str]:
    return _WORD_RE.findall(text)


# --------------------------------------------------------------------------- #
# Surface / lexical features
# --------------------------------------------------------------------------- #
def surface_features(text: str) -> dict[str, float]:
    """Punctuation, word-shape and lexical-diversity measurements.

    Rates are per-100-words unless the name says ``ratio`` (a proportion in
    [0, 1]), which keeps the values on comparable scales before standardisation.
    """
    words = words_of(text)
    lower = [w.lower() for w in words]
    n_words = len(words)
    n_chars = len(text)
    counts = Counter(lower)
    word_lengths = [len(w) for w in words]

    low = text.lower()
    contraction_hits = len(_CONTRACTION_RE.findall(text))
    llm_hits = sum(low.count(p) for p in LLM_REGISTER_PHRASES)
    transition_phrase_hits = sum(low.count(p) for p in TRANSITION_PHRASES)
    colloquial_hits = sum(1 for w in lower if w in _COLLOQUIAL_WORDS) + sum(
        low.count(p) for p in _COLLOQUIAL_PHRASES
    )
    hedge_hits = sum(1 for w in lower if w in _HEDGE_WORDS) + sum(
        low.count(p) for p in _HEDGE_PHRASES
    )

    first_word = lower[0] if lower else ""
    n_commas = text.count(",")
    stop_count = sum(1 for w in lower if w in STOPWORDS)
    punctuation = sum(1 for ch in text if ch in ",.;:!?-–-()[]\"'…")

    return {
        "sty_n_words": float(n_words),
        "sty_n_chars": float(n_chars),
        "sty_mean_word_len": _mean(word_lengths),
        "sty_std_word_len": _std(word_lengths),
        "sty_max_word_len": float(max(word_lengths)) if word_lengths else 0.0,
        "sty_long_word_ratio": _rate(sum(1 for n in word_lengths if n >= 7), n_words),
        "sty_short_word_ratio": _rate(sum(1 for n in word_lengths if n <= 3), n_words),
        "sty_syllables_per_word": _mean(count_syllables(w) for w in words),
        "sty_ttr": _rate(len(counts), n_words),
        # Guiraud's root TTR: far less length-dependent than raw TTR, which
        # matters because sentences differ in length by an order of magnitude.
        "sty_root_ttr": (len(counts) / math.sqrt(n_words)) if n_words else 0.0,
        "sty_hapax_ratio": _rate(sum(1 for c in counts.values() if c == 1), n_words),
        "sty_stopword_ratio": _rate(stop_count, n_words),
        "sty_content_word_ratio": _rate(n_words - stop_count, n_words),
        "sty_punct_density": _rate(punctuation, n_words, 100),
        "sty_comma_rate": _rate(n_commas, n_words, 100),
        "sty_semicolon_rate": _rate(text.count(";"), n_words, 100),
        "sty_colon_rate": _rate(text.count(":"), n_words, 100),
        "sty_dash_rate": _rate(text.count("-") + text.count("–") + len(re.findall(r"\s-\s", text)), n_words, 100),
        "sty_paren_rate": _rate(text.count("(") + text.count("["), n_words, 100),
        "sty_quote_rate": _rate(text.count('"') + text.count("“") + text.count("”"), n_words, 100),
        "sty_exclaim_rate": _rate(text.count("!"), n_words, 100),
        "sty_question_rate": _rate(text.count("?"), n_words, 100),
        "sty_ellipsis_rate": _rate(text.count("...") + text.count("…"), n_words, 100),
        "sty_digit_ratio": _rate(sum(1 for ch in text if ch.isdigit()), max(n_chars, 1)),
        "sty_uppercase_ratio": _rate(sum(1 for ch in text if ch.isupper()), max(n_chars, 1)),
        "sty_capitalised_word_ratio": _rate(sum(1 for w in words if w[:1].isupper()), n_words),
        "sty_contraction_rate": _rate(contraction_hits, n_words, 100),
        "sty_colloquial_rate": _rate(colloquial_hits, n_words, 100),
        "sty_llm_phrase_rate": _rate(llm_hits, n_words, 100),
        "sty_transition_word_rate": _rate(
            sum(1 for w in lower if w in _TRANSITION_WORD_SET), n_words, 100
        ),
        "sty_transition_phrase_rate": _rate(transition_phrase_hits, n_words, 100),
        "sty_sentence_initial_transition": 1.0 if first_word in _TRANSITION_WORD_SET else 0.0,
        "sty_hedge_rate": _rate(hedge_hits, n_words, 100),
        "sty_intensifier_rate": _rate(
            sum(1 for w in lower if w in _INTENSIFIER_SET), n_words, 100
        ),
        "sty_first_person_rate": _rate(sum(1 for w in lower if w in FIRST_PERSON), n_words, 100),
        "sty_nominalization_rate": _rate(
            sum(1 for w in lower if len(w) > 5 and w.endswith(NOMINALIZATION_SUFFIXES)),
            n_words,
            100,
        ),
        "sty_avg_commas_per_clause": float(n_commas),
    }


# --------------------------------------------------------------------------- #
# Syntactic features
# --------------------------------------------------------------------------- #
def _dep_depth(token) -> int:  # noqa: ANN001
    """Distance from ``token`` up to its sentence root.

    Note the ``.i`` comparison: spaCy builds a fresh ``Token`` object on each
    ``.head`` access, so ``node.head is node`` is *never* true even at the root.
    Identity comparison here silently pins every depth at the loop guard.
    """
    depth = 0
    node = token
    while node.head.i != node.i and depth < 64:
        node = node.head
        depth += 1
    return depth


def syntactic_features(span, text: str | None = None) -> dict[str, float]:  # noqa: ANN001
    """POS/dependency distributions and construction counts.

    ``span`` is a spaCy ``Span``/``Doc``. When it is ``None`` (spaCy
    unavailable) every syntactic feature is returned as 0.0 so that the vector
    keeps a stable width; the degraded state is reported separately via
    ``segmentation_backend``.
    """
    features: dict[str, float] = dict.fromkeys(SYN_FEATURE_NAMES, 0.0)
    if span is None:
        if text:
            features["syn_ends_with_period"] = 1.0 if text.rstrip().endswith(".") else 0.0
        return features

    tokens = [t for t in span if not t.is_space]
    if not tokens:
        return features

    real = [t for t in tokens if not t.is_punct]
    n_real = len(real) or 1

    pos_counts = Counter(t.pos_ for t in tokens)
    dep_counts = Counter(t.dep_ for t in tokens)

    for tag in POS_TAGS:
        features[f"syn_pos_{tag.lower()}"] = _rate(pos_counts.get(tag, 0), len(tokens))
    for dep in DEP_LABELS:
        features[f"syn_dep_{dep}"] = _rate(dep_counts.get(dep, 0), len(tokens))

    depths = [_dep_depth(t) for t in real]
    # Same caveat as _dep_depth: compare token indices, not object identity.
    distances = [abs(t.i - t.head.i) for t in real if t.head.i != t.i]

    n_clauses = sum(1 for t in tokens if t.dep_ in CLAUSAL_DEPS) + sum(
        1 for t in tokens if t.dep_ == "ROOT" and t.pos_ in {"VERB", "AUX"}
    )
    n_verbs = pos_counts.get("VERB", 0) + pos_counts.get("AUX", 0)
    n_nouns = pos_counts.get("NOUN", 0) + pos_counts.get("PROPN", 0)

    first = tokens[0]
    features.update(
        {
            "syn_n_clauses": float(n_clauses),
            "syn_clauses_per_sentence": float(n_clauses),
            "syn_max_dep_depth": float(max(depths)) if depths else 0.0,
            "syn_mean_dep_depth": _mean(depths),
            "syn_mean_dep_distance": _mean(distances),
            "syn_max_dep_distance": float(max(distances)) if distances else 0.0,
            "syn_passive_ratio": _rate(
                sum(1 for t in tokens if t.dep_ in PASSIVE_DEPS), n_real
            ),
            "syn_subordinate_ratio": _rate(pos_counts.get("SCONJ", 0), n_real),
            "syn_coordination_ratio": _rate(pos_counts.get("CCONJ", 0), n_real),
            "syn_noun_verb_ratio": (n_nouns / n_verbs) if n_verbs else float(n_nouns),
            "syn_adj_noun_ratio": (pos_counts.get("ADJ", 0) / n_nouns) if n_nouns else 0.0,
            "syn_adverb_verb_ratio": (pos_counts.get("ADV", 0) / n_verbs) if n_verbs else 0.0,
            "syn_function_word_ratio": _rate(
                sum(1 for t in real if t.text.lower() in STOPWORDS), n_real
            ),
            "syn_pos_entropy": _entropy(pos_counts.values()),
            "syn_dep_entropy": _entropy(dep_counts.values()),
            "syn_starts_with_pronoun": 1.0 if first.pos_ == "PRON" else 0.0,
            "syn_starts_with_conj": 1.0 if first.pos_ in {"CCONJ", "SCONJ"} else 0.0,
            "syn_starts_with_adverb": 1.0 if first.pos_ == "ADV" else 0.0,
            "syn_starts_with_prep": 1.0 if first.pos_ == "ADP" else 0.0,
            "syn_ends_with_period": 1.0 if tokens[-1].text == "." else 0.0,
        }
    )
    return features


# --------------------------------------------------------------------------- #
# Distributions used for style-shift comparisons
# --------------------------------------------------------------------------- #
def pos_distribution(span, text: str | None = None) -> dict[str, float]:  # noqa: ANN001
    """Normalised POS distribution over the fixed tag inventory."""
    dist = dict.fromkeys(POS_TAGS, 0.0)
    if span is None:
        return dist
    tokens = [t for t in span if not t.is_space]
    if not tokens:
        return dist
    counts = Counter(t.pos_ for t in tokens)
    total = sum(counts.get(tag, 0) for tag in POS_TAGS) or 1
    for tag in POS_TAGS:
        dist[tag] = counts.get(tag, 0) / total
    return dist


def function_word_profile(text: str) -> dict[str, float]:
    """Relative frequency of each function word (per word, not per function word)."""
    words = [w.lower() for w in words_of(text)]
    total = len(words) or 1
    counts = Counter(w for w in words if w in _FUNCTION_WORD_INDEX)
    return {w: counts.get(w, 0) / total for w in FUNCTION_WORDS}


def pos_sequence(span) -> list[str]:  # noqa: ANN001
    if span is None:
        return []
    return [t.pos_ for t in span if not t.is_space]


def token_views(span, text: str) -> tuple[list[str], list[str]]:  # noqa: ANN001
    """``(all_tokens_lowercased, content_words)`` for repetition analysis."""
    if span is not None:
        tokens = [t.text.lower() for t in span if not t.is_space and not t.is_punct]
        content = [
            t.text.lower()
            for t in span
            if not t.is_space and not t.is_punct and t.text.lower() not in STOPWORDS
        ]
        return tokens, content
    tokens = [w.lower() for w in words_of(text)]
    return tokens, [w for w in tokens if w not in STOPWORDS]


def measure_sentence(sentence, /) -> None:
    """Populate the stylometric/syntactic slots of a :class:`SentenceUnit`."""
    text = sentence.text
    sty = surface_features(text)
    syn = syntactic_features(sentence.span, text)

    # Commas per clause needs both layers, so it is finalised here.
    n_clauses = syn.get("syn_n_clauses", 0.0) or 1.0
    sty["sty_avg_commas_per_clause"] = text.count(",") / n_clauses

    sentence.stylometry = sty
    sentence.syntax = syn
    sentence.pos_distribution = pos_distribution(sentence.span, text)
    sentence.function_word_profile = function_word_profile(text)
    sentence.pos_sequence = pos_sequence(sentence.span)
    sentence.tokens, sentence.content_words = token_views(sentence.span, text)
