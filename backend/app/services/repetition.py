"""Repetition analysis: lexical n-grams, syntactic templates, discourse patterns.

Three kinds of repetition are measured separately because they mean different
things:

* **Lexical n-gram repetition** - the same words in the same order. Common in
  drafts written under time pressure *and* in machine text that reuses a phrase
  bank, so on its own it is ambiguous.
* **Syntactic template repetition** - the same POS skeleton across sentences
  (e.g. a run of ``PRON VERB DET ADJ NOUN``). This is the signal most associated
  with generated text, and it survives paraphrasing.
* **Discourse-pattern repetition** - repeated sentence openers and connectives.

Feature prefix: ``rep_``. Concrete repeated spans are also returned for the
explanation engine, so the UI can name the exact phrase that triggered a flag.
"""

from __future__ import annotations

import math
from collections import Counter

from app.services.lexicons import (
    TRANSITION_PHRASES,
    TRANSITION_WORDS,
    UNINFORMATIVE_NGRAM_WORDS,
)

REP_FEATURE_NAMES: tuple[str, ...] = (
    "rep_bigram_repeat_ratio",
    "rep_trigram_repeat_ratio",
    "rep_fourgram_repeat_ratio",
    "rep_fivegram_repeat_ratio",
    "rep_distinct2",
    "rep_distinct3",
    "rep_distinct4",
    "rep_max_ngram_repeat_count",
    "rep_repeated_ngram_types",
    "rep_content_word_repeat_ratio",
    "rep_top_content_word_share",
    "rep_content_word_entropy",
    "rep_pos_trigram_repeat_ratio",
    "rep_pos_fourgram_repeat_ratio",
    "rep_pos_distinct4",
    "rep_max_pos_template_count",
    "rep_sentence_opener_repeat_ratio",
    "rep_max_opener_count",
    "rep_transition_word_types",
    "rep_transition_phrase_count",
    "rep_max_sentence_jaccard",
    "rep_mean_sentence_jaccard",
    "rep_frac_sentence_pairs_over_30pct",
    "rep_paragraph_opener_repeat_ratio",
)

MIN_TOKENS_FOR_NGRAMS = 12


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _repeat_ratio(tokens: list[str], n: int) -> float:
    """Fraction of ``n``-gram *instances* that are not the first occurrence."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated_instances = sum(c - 1 for c in counts.values() if c > 1)
    return repeated_instances / len(grams)


def _distinct(tokens: list[str], n: int) -> float:
    """distinct-n: unique n-gram types divided by n-gram instances."""
    grams = _ngrams(tokens, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def _entropy(counts) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counts if c > 0))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def repeated_spans(document, min_n: int = 3, top_k: int = 8) -> list[dict[str, object]]:  # noqa: ANN001
    """Concrete repeated word sequences, for display as evidence.

    Only n-grams carrying at least one informative word are reported. A run like
    "is one of the" recurring three times is a property of English, not of this
    author, and showing it as evidence would dilute the real findings. The filter
    uses :data:`UNINFORMATIVE_NGRAM_WORDS` rather than the feature-space function
    word list, so it can be tuned without invalidating trained artifacts.
    """
    tokens: list[str] = []
    owners: list[int] = []
    for sentence in document.sentences:
        for token in sentence.tokens:
            tokens.append(token)
            owners.append(sentence.index)

    found: dict[tuple[str, ...], list[int]] = {}
    for n in (5, 4, 3):
        if n < min_n:
            continue
        counts: dict[tuple[str, ...], list[int]] = {}
        for i, gram in enumerate(_ngrams(tokens, n)):
            counts.setdefault(gram, []).append(i)
        for gram, positions in counts.items():
            if len(positions) < 2:
                continue
            if all(word in UNINFORMATIVE_NGRAM_WORDS for word in gram):
                continue
            # Skip an n-gram that is entirely contained in an already reported
            # longer one (avoids reporting "a testament to my" and "testament to my").
            if any(_is_sub(gram, longer) for longer in found):
                continue
            found[gram] = positions

    results = [
        {
            "phrase": " ".join(gram),
            "length": len(gram),
            "count": len(positions),
            "sentence_indices": sorted({owners[p] for p in positions if p < len(owners)}),
        }
        for gram, positions in found.items()
    ]
    results.sort(key=lambda r: (-int(r["length"]), -int(r["count"])))
    return results[:top_k]


def _is_sub(short: tuple[str, ...], long: tuple[str, ...]) -> bool:
    if len(short) >= len(long):
        return False
    return any(long[i : i + len(short)] == short for i in range(len(long) - len(short) + 1))


def repeated_pos_templates(document, n: int = 4, top_k: int = 6) -> list[dict[str, object]]:  # noqa: ANN001
    """Repeated POS skeletons across *different* sentences."""
    per_sentence: dict[tuple[str, ...], set[int]] = {}
    for sentence in document.sentences:
        seq = [p for p in sentence.pos_sequence if p != "PUNCT"]
        for gram in set(_ngrams(seq, n)):
            per_sentence.setdefault(gram, set()).add(sentence.index)

    results = [
        {
            "template": " ".join(gram),
            "sentence_count": len(indices),
            "sentence_indices": sorted(indices),
        }
        for gram, indices in per_sentence.items()
        if len(indices) >= 3
    ]
    results.sort(key=lambda r: -int(r["sentence_count"]))
    return results[:top_k]


def extract(document) -> dict[str, float]:  # noqa: ANN001
    """Document-level repetition features."""
    features = dict.fromkeys(REP_FEATURE_NAMES, 0.0)
    sentences = document.sentences
    if not sentences:
        return features

    tokens: list[str] = []
    content: list[str] = []
    pos_seq: list[str] = []
    for sentence in sentences:
        tokens.extend(sentence.tokens)
        content.extend(sentence.content_words)
        pos_seq.extend(p for p in sentence.pos_sequence if p != "PUNCT")

    if len(tokens) >= MIN_TOKENS_FOR_NGRAMS:
        all_counts = Counter()
        for n in (2, 3, 4, 5):
            all_counts.update(Counter(_ngrams(tokens, n)))
        repeated_types = sum(1 for g, c in all_counts.items() if c > 1 and len(g) >= 3)
        features.update(
            {
                "rep_bigram_repeat_ratio": _repeat_ratio(tokens, 2),
                "rep_trigram_repeat_ratio": _repeat_ratio(tokens, 3),
                "rep_fourgram_repeat_ratio": _repeat_ratio(tokens, 4),
                "rep_fivegram_repeat_ratio": _repeat_ratio(tokens, 5),
                "rep_distinct2": _distinct(tokens, 2),
                "rep_distinct3": _distinct(tokens, 3),
                "rep_distinct4": _distinct(tokens, 4),
                "rep_max_ngram_repeat_count": float(
                    max((c for g, c in all_counts.items() if len(g) >= 3), default=0)
                ),
                "rep_repeated_ngram_types": float(repeated_types),
            }
        )

    if content:
        content_counts = Counter(content)
        features["rep_content_word_repeat_ratio"] = sum(
            c - 1 for c in content_counts.values() if c > 1
        ) / len(content)
        features["rep_top_content_word_share"] = content_counts.most_common(1)[0][1] / len(content)
        features["rep_content_word_entropy"] = _entropy(content_counts.values())

    if pos_seq:
        pos_counts = Counter(_ngrams(pos_seq, 4))
        features.update(
            {
                "rep_pos_trigram_repeat_ratio": _repeat_ratio(pos_seq, 3),
                "rep_pos_fourgram_repeat_ratio": _repeat_ratio(pos_seq, 4),
                "rep_pos_distinct4": _distinct(pos_seq, 4),
                "rep_max_pos_template_count": float(
                    max(pos_counts.values(), default=0)
                ),
            }
        )

    # --- discourse patterns -------------------------------------------------
    openers = [
        " ".join(sentence.tokens[:2]) for sentence in sentences if len(sentence.tokens) >= 2
    ]
    if openers:
        opener_counts = Counter(openers)
        features["rep_sentence_opener_repeat_ratio"] = sum(
            c - 1 for c in opener_counts.values() if c > 1
        ) / len(openers)
        features["rep_max_opener_count"] = float(max(opener_counts.values()))

    low = document.text.lower()
    features["rep_transition_word_types"] = float(
        sum(1 for w in TRANSITION_WORDS if f" {w} " in f" {low} " or low.startswith(w))
    )
    features["rep_transition_phrase_count"] = float(
        sum(low.count(p) for p in TRANSITION_PHRASES)
    )

    # --- inter-sentence similarity -----------------------------------------
    sets = [set(s.content_words) for s in sentences if s.content_words]
    if len(sets) >= 2:
        scores = [
            _jaccard(sets[i], sets[j])
            for i in range(len(sets))
            for j in range(i + 1, len(sets))
        ]
        if scores:
            features["rep_max_sentence_jaccard"] = float(max(scores))
            features["rep_mean_sentence_jaccard"] = float(sum(scores) / len(scores))
            features["rep_frac_sentence_pairs_over_30pct"] = float(
                sum(1 for s in scores if s > 0.3) / len(scores)
            )

    paragraph_openers = []
    for para in document.paragraphs:
        if para.sentence_indices:
            first = sentences[para.sentence_indices[0]]
            if first.tokens:
                paragraph_openers.append(first.tokens[0])
    if len(paragraph_openers) >= 2:
        counts = Counter(paragraph_openers)
        features["rep_paragraph_opener_repeat_ratio"] = sum(
            c - 1 for c in counts.values() if c > 1
        ) / len(paragraph_openers)

    return features
