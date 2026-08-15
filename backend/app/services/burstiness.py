"""Sentence rhythm / burstiness analysis.

Human prose tends to be *bursty*: long sentences next to short ones, uneven
punctuation, variable complexity. Machine prose tends to regress toward a
comfortable middle length. This module quantifies that variation.

Nothing here decides anything. Low variation is one feature among ~250; the
classifier learns how much it is worth, and the explanation engine always
reports it relative to the training distribution rather than as a verdict.

Feature prefix: ``bur_``
"""

from __future__ import annotations

import math
from collections import Counter

BUR_FEATURE_NAMES: tuple[str, ...] = (
    "bur_mean_sent_len",
    "bur_median_sent_len",
    "bur_std_sent_len",
    "bur_cv_sent_len",
    "bur_iqr_sent_len",
    "bur_range_sent_len",
    "bur_min_sent_len",
    "bur_max_sent_len",
    "bur_burstiness_index",
    "bur_sent_len_entropy",
    "bur_normalised_len_entropy",
    "bur_mean_abs_adjacent_diff",
    "bur_std_abs_adjacent_diff",
    "bur_max_abs_adjacent_diff",
    "bur_mean_rel_adjacent_diff",
    "bur_lag1_autocorr_sent_len",
    "bur_frac_sent_within_20pct_of_mean",
    "bur_frac_sent_len_10_25",
    "bur_direction_changes",
    "bur_cv_word_len",
    "bur_cv_punct_density",
    "bur_cv_comma_rate",
    "bur_cv_ttr",
    "bur_cv_root_ttr",
    "bur_cv_clauses",
    "bur_cv_dep_depth",
    "bur_cv_logprob",
    "bur_cv_perplexity",
    "bur_std_logprob_across_sentences",
    "bur_mean_abs_adjacent_logprob_diff",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _cv(values: list[float]) -> float:
    """Coefficient of variation - scale-free dispersion."""
    mu = _mean(values)
    if abs(mu) < 1e-9:
        return 0.0
    return _std(values) / abs(mu)


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else float((s[mid - 1] + s[mid]) / 2)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q / 100 * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


def length_entropy(lengths: list[int], bin_width: int = 5) -> tuple[float, float]:
    """Shannon entropy of the binned sentence-length distribution.

    Returns ``(entropy_bits, entropy_normalised_by_max)``. Binning at 5 words
    keeps the measure stable for short essays where every raw length would
    otherwise be unique (and entropy trivially maximal).
    """
    if not lengths:
        return 0.0, 0.0
    bins = Counter(int(n) // bin_width for n in lengths)
    total = sum(bins.values())
    h = 0.0
    for count in bins.values():
        p = count / total
        h -= p * math.log2(p)
    max_h = math.log2(len(bins)) if len(bins) > 1 else 0.0
    return float(h), float(h / max_h) if max_h > 0 else 0.0


def lag1_autocorrelation(values: list[float]) -> float:
    """Correlation between each value and the next.

    Near zero means lengths look independent (human-ish alternation);
    a strongly positive value means runs of similar lengths.
    """
    n = len(values)
    if n < 3:
        return 0.0
    mu = _mean(values)
    denom = sum((v - mu) ** 2 for v in values)
    if denom < 1e-12:
        return 0.0
    num = sum((values[i] - mu) * (values[i + 1] - mu) for i in range(n - 1))
    return float(max(-1.0, min(1.0, num / denom)))


def burstiness_index(values: list[float]) -> float:
    """Goh & Barabási burstiness: ``(sigma - mu) / (sigma + mu)`` in [-1, 1].

    -1 is perfectly regular, 0 is Poisson-like, and positive values are bursty.
    """
    mu, sigma = _mean(values), _std(values)
    if mu + sigma < 1e-9:
        return 0.0
    return float((sigma - mu) / (sigma + mu))


def direction_changes(values: list[float]) -> float:
    """Fraction of positions where the sentence-length trend reverses.

    Alternating long/short prose scores high; a monotone drift or a flat run
    scores low.
    """
    if len(values) < 3:
        return 0.0
    diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    changes = sum(
        1
        for i in range(len(diffs) - 1)
        if diffs[i] != 0 and diffs[i + 1] != 0 and (diffs[i] > 0) != (diffs[i + 1] > 0)
    )
    return float(changes / (len(diffs) - 1))


def sentence_rhythm(document) -> list[dict[str, float]]:  # noqa: ANN001
    """Per-sentence rhythm series for the frontend chart.

    Each entry carries the raw measurements the UI plots, so the chart can never
    disagree with the numbers in the evidence panel.
    """
    series: list[dict[str, float]] = []
    lengths = [float(s.n_words) for s in document.sentences]
    mean_len = _mean(lengths)
    for i, sentence in enumerate(document.sentences):
        n = float(sentence.n_words)
        series.append(
            {
                "index": i,
                "paragraph_index": sentence.paragraph_index,
                "words": n,
                "deviation_from_mean": round(n - mean_len, 2),
                "abs_diff_prev": round(abs(n - lengths[i - 1]), 2) if i > 0 else 0.0,
                "clauses": round(sentence.syntax.get("syn_n_clauses", 0.0), 2),
                "mean_logprob": round(sentence.lm.get("lm_mean_logprob", 0.0), 4),
                "perplexity": round(sentence.lm.get("lm_perplexity", 0.0), 2),
            }
        )
    return series


def extract(document) -> dict[str, float]:  # noqa: ANN001
    """Document-level burstiness features."""
    features = dict.fromkeys(BUR_FEATURE_NAMES, 0.0)
    sentences = document.sentences
    if not sentences:
        return features

    lengths = [float(s.n_words) for s in sentences]
    n = len(lengths)
    mean_len = _mean(lengths)

    adjacent = [abs(lengths[i + 1] - lengths[i]) for i in range(n - 1)]
    relative = [
        abs(lengths[i + 1] - lengths[i]) / max(lengths[i], 1.0) for i in range(n - 1)
    ]

    h, h_norm = length_entropy([int(v) for v in lengths])

    word_lens = [s.stylometry.get("sty_mean_word_len", 0.0) for s in sentences]
    punct = [s.stylometry.get("sty_punct_density", 0.0) for s in sentences]
    commas = [s.stylometry.get("sty_comma_rate", 0.0) for s in sentences]
    ttr = [s.stylometry.get("sty_ttr", 0.0) for s in sentences]
    root_ttr = [s.stylometry.get("sty_root_ttr", 0.0) for s in sentences]
    clauses = [s.syntax.get("syn_n_clauses", 0.0) for s in sentences]
    depth = [s.syntax.get("syn_mean_dep_depth", 0.0) for s in sentences]
    logprob = [s.lm.get("lm_mean_logprob", 0.0) for s in sentences if s.lm]
    perplexity = [s.lm.get("lm_perplexity", 0.0) for s in sentences if s.lm]

    features.update(
        {
            "bur_mean_sent_len": mean_len,
            "bur_median_sent_len": _median(lengths),
            "bur_std_sent_len": _std(lengths),
            "bur_cv_sent_len": _cv(lengths),
            "bur_iqr_sent_len": _percentile(lengths, 75) - _percentile(lengths, 25),
            "bur_range_sent_len": float(max(lengths) - min(lengths)),
            "bur_min_sent_len": float(min(lengths)),
            "bur_max_sent_len": float(max(lengths)),
            "bur_burstiness_index": burstiness_index(lengths),
            "bur_sent_len_entropy": h,
            "bur_normalised_len_entropy": h_norm,
            "bur_mean_abs_adjacent_diff": _mean(adjacent),
            "bur_std_abs_adjacent_diff": _std(adjacent),
            "bur_max_abs_adjacent_diff": float(max(adjacent)) if adjacent else 0.0,
            "bur_mean_rel_adjacent_diff": _mean(relative),
            "bur_lag1_autocorr_sent_len": lag1_autocorrelation(lengths),
            "bur_frac_sent_within_20pct_of_mean": (
                sum(1 for v in lengths if abs(v - mean_len) <= 0.2 * max(mean_len, 1.0)) / n
            ),
            "bur_frac_sent_len_10_25": sum(1 for v in lengths if 10 <= v <= 25) / n,
            "bur_direction_changes": direction_changes(lengths),
            "bur_cv_word_len": _cv(word_lens),
            "bur_cv_punct_density": _cv(punct),
            "bur_cv_comma_rate": _cv(commas),
            "bur_cv_ttr": _cv(ttr),
            "bur_cv_root_ttr": _cv(root_ttr),
            "bur_cv_clauses": _cv(clauses),
            "bur_cv_dep_depth": _cv(depth),
            "bur_cv_logprob": _cv(logprob),
            "bur_cv_perplexity": _cv(perplexity),
            "bur_std_logprob_across_sentences": _std(logprob),
            "bur_mean_abs_adjacent_logprob_diff": _mean(
                [abs(logprob[i + 1] - logprob[i]) for i in range(len(logprob) - 1)]
            ),
        }
    )
    return features
