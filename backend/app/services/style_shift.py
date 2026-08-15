"""Within-essay style shift detection.

The most useful question for the realistic "human wrote it, AI edited part of it"
case is not *does this look like AI?* but *does this look like the rest of this
essay?* Every author has a personal baseline; a passage that departs sharply from
that baseline is worth surfacing.

Two feature families are produced:

``ctx_``   per-sentence deviation from the author's own document baseline
``shift_`` document-level dispersion and change-point statistics

An important caveat, stated in the UI as well as here: a style shift is evidence
of *editing or register change*, not of authorship. Quoting a source, switching
from narrative to reflection, or simply writing a stronger conclusion all cause
genuine shifts in human writing.
"""

from __future__ import annotations

import math

from app.services.corpus_analyzer import cosine, js_divergence
from app.services.lexicons import FUNCTION_WORDS, POS_TAGS

CTX_FEATURE_NAMES: tuple[str, ...] = (
    "ctx_rel_position",
    "ctx_doc_log_n_sentences",
    "ctx_is_paragraph_first",
    "ctx_is_paragraph_last",
    "ctx_z_n_words",
    "ctx_z_mean_word_len",
    "ctx_z_ttr",
    "ctx_z_root_ttr",
    "ctx_z_punct_density",
    "ctx_z_comma_rate",
    "ctx_z_stopword_ratio",
    "ctx_z_mean_logprob",
    "ctx_z_log_perplexity",
    "ctx_z_mean_entropy",
    "ctx_z_frac_top1",
    "ctx_z_mean_log_rank",
    "ctx_z_clauses",
    "ctx_z_dep_depth",
    "ctx_z_first_person_rate",
    "ctx_z_nominalization_rate",
    "ctx_abs_len_diff_prev",
    "ctx_abs_len_diff_next",
    "ctx_rel_len_diff_prev",
    "ctx_logprob_diff_prev",
    "ctx_logprob_diff_next",
    "ctx_pos_js_to_doc",
    "ctx_pos_cosine_to_doc",
    "ctx_funcword_cosine_to_doc",
    "ctx_pos_js_to_prev",
    "ctx_pos_js_to_paragraph",
    "ctx_style_distance_to_doc",
    "ctx_style_distance_to_prev",
    "ctx_max_neighbour_style_distance",
)

SHIFT_FEATURE_NAMES: tuple[str, ...] = (
    "shift_max_abs_z_logprob",
    "shift_mean_abs_z_logprob",
    "shift_frac_sent_abs_z_logprob_gt2",
    "shift_max_abs_z_len",
    "shift_frac_sent_abs_z_len_gt2",
    "shift_mean_pos_js_to_doc",
    "shift_max_pos_js_to_doc",
    "shift_std_pos_js_to_doc",
    "shift_mean_style_distance_to_doc",
    "shift_max_style_distance_to_doc",
    "shift_std_style_distance_to_doc",
    "shift_mean_adjacent_style_distance",
    "shift_max_adjacent_style_distance",
    "shift_n_changepoints",
    "shift_max_segment_mean_gap",
    "shift_paragraph_logprob_std",
    "shift_paragraph_logprob_range",
    "shift_paragraph_len_cv",
    "shift_paragraph_ttr_std",
    "shift_paragraph_style_distance_max",
    "shift_paragraph_count_with_shift",
)

# Feature subset used as the "style vector" for distance comparisons. These are
# register-bearing and roughly topic-free.
_STYLE_KEYS: tuple[str, ...] = (
    "sty_mean_word_len",
    "sty_root_ttr",
    "sty_stopword_ratio",
    "sty_punct_density",
    "sty_comma_rate",
    "sty_contraction_rate",
    "sty_first_person_rate",
    "sty_nominalization_rate",
    "sty_hedge_rate",
    "sty_transition_word_rate",
    "sty_syllables_per_word",
    "sty_long_word_ratio",
)
_SYN_STYLE_KEYS: tuple[str, ...] = (
    "syn_mean_dep_depth",
    "syn_mean_dep_distance",
    "syn_subordinate_ratio",
    "syn_coordination_ratio",
    "syn_passive_ratio",
    "syn_noun_verb_ratio",
    "syn_pos_entropy",
)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = _mean(values)
    return float(math.sqrt(sum((v - mu) ** 2 for v in values) / len(values)))


def _cv(values: list[float]) -> float:
    mu = _mean(values)
    return _std(values) / abs(mu) if abs(mu) > 1e-9 else 0.0


def _z(value: float, mu: float, sigma: float) -> float:
    """Robustly clipped z-score.

    Clipping at +/-6 keeps a single outlier sentence from dominating the scaler
    that the classifier sees.
    """
    if sigma < 1e-9:
        return 0.0
    return float(max(-6.0, min(6.0, (value - mu) / sigma)))


def _style_vector(sentence) -> list[float]:  # noqa: ANN001
    return [sentence.stylometry.get(k, 0.0) for k in _STYLE_KEYS] + [
        sentence.syntax.get(k, 0.0) for k in _SYN_STYLE_KEYS
    ]


def _normalised_distance(a: list[float], b: list[float], scales: list[float]) -> float:
    """Mean absolute difference scaled by each dimension's document-level spread.

    Scaling per dimension matters: raw Euclidean distance would be dominated by
    ``syn_mean_dep_distance`` (units of tokens) and ignore ``sty_root_ttr``
    (units of ~1).
    """
    if not a or not b:
        return 0.0
    total = 0.0
    for x, y, s in zip(a, b, scales, strict=False):
        total += abs(x - y) / s if s > 1e-9 else 0.0
    return float(total / len(a))


def annotate(document) -> dict[str, float]:  # noqa: ANN001
    """Fill each sentence's ``ctx_*`` block and return the ``shift_*`` block.

    Must run after stylometry and LM scoring, since it compares those values.
    """
    sentences = document.sentences
    shift = dict.fromkeys(SHIFT_FEATURE_NAMES, 0.0)
    if not sentences:
        return shift

    n = len(sentences)
    lengths = [float(s.n_words) for s in sentences]

    def series(getter) -> list[float]:  # noqa: ANN001
        return [getter(s) for s in sentences]

    tracked = {
        "n_words": lengths,
        "mean_word_len": series(lambda s: s.stylometry.get("sty_mean_word_len", 0.0)),
        "ttr": series(lambda s: s.stylometry.get("sty_ttr", 0.0)),
        "root_ttr": series(lambda s: s.stylometry.get("sty_root_ttr", 0.0)),
        "punct_density": series(lambda s: s.stylometry.get("sty_punct_density", 0.0)),
        "comma_rate": series(lambda s: s.stylometry.get("sty_comma_rate", 0.0)),
        "stopword_ratio": series(lambda s: s.stylometry.get("sty_stopword_ratio", 0.0)),
        "mean_logprob": series(lambda s: s.lm.get("lm_mean_logprob", 0.0)),
        "log_perplexity": series(lambda s: s.lm.get("lm_log_perplexity", 0.0)),
        "mean_entropy": series(lambda s: s.lm.get("lm_mean_entropy", 0.0)),
        "frac_top1": series(lambda s: s.lm.get("lm_frac_top1", 0.0)),
        "mean_log_rank": series(lambda s: s.lm.get("lm_mean_log_rank", 0.0)),
        "clauses": series(lambda s: s.syntax.get("syn_n_clauses", 0.0)),
        "dep_depth": series(lambda s: s.syntax.get("syn_mean_dep_depth", 0.0)),
        "first_person_rate": series(lambda s: s.stylometry.get("sty_first_person_rate", 0.0)),
        "nominalization_rate": series(
            lambda s: s.stylometry.get("sty_nominalization_rate", 0.0)
        ),
    }
    stats = {k: (_mean(v), _std(v)) for k, v in tracked.items()}

    # Document-level reference distributions.
    doc_pos = _aggregate_pos(sentences)
    doc_funcword = _aggregate_profile(
        [s.function_word_profile for s in sentences], FUNCTION_WORDS
    )
    style_vectors = [_style_vector(s) for s in sentences]
    doc_style = [
        _mean([vec[i] for vec in style_vectors]) for i in range(len(style_vectors[0]))
    ]
    style_scales = [
        max(_std([vec[i] for vec in style_vectors]), abs(doc_style[i]) * 0.1, 1e-6)
        for i in range(len(doc_style))
    ]

    paragraph_pos = {
        para.index: _aggregate_pos([sentences[i] for i in para.sentence_indices])
        for para in document.paragraphs
    }

    logprob_series = tracked["mean_logprob"]
    style_to_doc: list[float] = []
    adjacent_style: list[float] = []
    pos_js_to_doc: list[float] = []

    for i, sentence in enumerate(sentences):
        para = document.paragraphs[sentence.paragraph_index] if document.paragraphs else None
        prev_len = lengths[i - 1] if i > 0 else lengths[i]
        next_len = lengths[i + 1] if i < n - 1 else lengths[i]
        prev_lp = logprob_series[i - 1] if i > 0 else logprob_series[i]
        next_lp = logprob_series[i + 1] if i < n - 1 else logprob_series[i]

        js_doc = js_divergence(sentence.pos_distribution, doc_pos, POS_TAGS)
        pos_js_to_doc.append(js_doc)
        js_prev = (
            js_divergence(sentence.pos_distribution, sentences[i - 1].pos_distribution, POS_TAGS)
            if i > 0
            else 0.0
        )
        js_para = (
            js_divergence(sentence.pos_distribution, paragraph_pos.get(sentence.paragraph_index, doc_pos), POS_TAGS)
            if para is not None
            else 0.0
        )

        d_doc = _normalised_distance(style_vectors[i], doc_style, style_scales)
        d_prev = (
            _normalised_distance(style_vectors[i], style_vectors[i - 1], style_scales)
            if i > 0
            else 0.0
        )
        d_next = (
            _normalised_distance(style_vectors[i], style_vectors[i + 1], style_scales)
            if i < n - 1
            else 0.0
        )
        style_to_doc.append(d_doc)
        if i > 0:
            adjacent_style.append(d_prev)

        context: dict[str, float] = {
            "ctx_rel_position": i / max(n - 1, 1),
            "ctx_doc_log_n_sentences": math.log1p(n),
            "ctx_is_paragraph_first": 1.0
            if para is not None and para.sentence_indices[:1] == [i]
            else 0.0,
            "ctx_is_paragraph_last": 1.0
            if para is not None and para.sentence_indices[-1:] == [i]
            else 0.0,
            "ctx_abs_len_diff_prev": abs(lengths[i] - prev_len),
            "ctx_abs_len_diff_next": abs(lengths[i] - next_len),
            "ctx_rel_len_diff_prev": abs(lengths[i] - prev_len) / max(prev_len, 1.0),
            "ctx_logprob_diff_prev": logprob_series[i] - prev_lp,
            "ctx_logprob_diff_next": logprob_series[i] - next_lp,
            "ctx_pos_js_to_doc": js_doc,
            "ctx_pos_cosine_to_doc": cosine(sentence.pos_distribution, doc_pos, POS_TAGS),
            "ctx_funcword_cosine_to_doc": cosine(
                sentence.function_word_profile, doc_funcword, FUNCTION_WORDS
            ),
            "ctx_pos_js_to_prev": js_prev,
            "ctx_pos_js_to_paragraph": js_para,
            "ctx_style_distance_to_doc": d_doc,
            "ctx_style_distance_to_prev": d_prev,
            "ctx_max_neighbour_style_distance": max(d_prev, d_next),
        }
        for name, values in tracked.items():
            mu, sigma = stats[name]
            context[f"ctx_z_{name}"] = _z(values[i], mu, sigma)

        sentence.context = {k: context.get(k, 0.0) for k in CTX_FEATURE_NAMES}

    # ------------------------------------------------------- document level
    z_logprob = [abs(s.context.get("ctx_z_mean_logprob", 0.0)) for s in sentences]
    z_len = [abs(s.context.get("ctx_z_n_words", 0.0)) for s in sentences]
    changepoints, max_gap = _changepoints(logprob_series)

    para_stats = _paragraph_stats(document, style_vectors, style_scales, doc_style)

    shift.update(
        {
            "shift_max_abs_z_logprob": max(z_logprob),
            "shift_mean_abs_z_logprob": _mean(z_logprob),
            "shift_frac_sent_abs_z_logprob_gt2": sum(1 for v in z_logprob if v > 2.0) / n,
            "shift_max_abs_z_len": max(z_len),
            "shift_frac_sent_abs_z_len_gt2": sum(1 for v in z_len if v > 2.0) / n,
            "shift_mean_pos_js_to_doc": _mean(pos_js_to_doc),
            "shift_max_pos_js_to_doc": max(pos_js_to_doc),
            "shift_std_pos_js_to_doc": _std(pos_js_to_doc),
            "shift_mean_style_distance_to_doc": _mean(style_to_doc),
            "shift_max_style_distance_to_doc": max(style_to_doc),
            "shift_std_style_distance_to_doc": _std(style_to_doc),
            "shift_mean_adjacent_style_distance": _mean(adjacent_style),
            "shift_max_adjacent_style_distance": max(adjacent_style) if adjacent_style else 0.0,
            "shift_n_changepoints": float(changepoints),
            "shift_max_segment_mean_gap": max_gap,
            **para_stats,
        }
    )
    return shift


def _aggregate_pos(sentences) -> dict[str, float]:  # noqa: ANN001
    totals = dict.fromkeys(POS_TAGS, 0.0)
    weight = 0.0
    for sentence in sentences:
        n_tokens = max(len(sentence.pos_sequence), 1)
        for tag in POS_TAGS:
            totals[tag] += sentence.pos_distribution.get(tag, 0.0) * n_tokens
        weight += n_tokens
    if weight <= 0:
        return totals
    return {k: v / weight for k, v in totals.items()}


def _aggregate_profile(profiles, keys) -> dict[str, float]:  # noqa: ANN001
    if not profiles:
        return dict.fromkeys(keys, 0.0)
    return {k: _mean([p.get(k, 0.0) for p in profiles]) for k in keys}


def _changepoints(series: list[float], threshold: float = 1.5) -> tuple[int, float]:
    """Count positions where the running mean shifts by more than ``threshold``
    standard deviations, using a simple two-window scan.

    This is deliberately a transparent statistic rather than a learned segmenter:
    the evidence panel has to be able to say *why* a boundary was reported.
    """
    n = len(series)
    if n < 6:
        return 0, 0.0
    sigma = _std(series)
    if sigma < 1e-9:
        return 0, 0.0
    window = max(2, min(4, n // 3))
    count = 0
    max_gap = 0.0
    for i in range(window, n - window + 1):
        left = _mean(series[i - window : i])
        right = _mean(series[i : i + window])
        gap = abs(right - left) / sigma
        max_gap = max(max_gap, gap)
        if gap > threshold:
            count += 1
    return count, float(max_gap)


def _paragraph_stats(document, style_vectors, style_scales, doc_style) -> dict[str, float]:  # noqa: ANN001
    logprobs: list[float] = []
    lengths: list[float] = []
    ttrs: list[float] = []
    distances: list[float] = []
    for para in document.paragraphs:
        members = [document.sentences[i] for i in para.sentence_indices]
        if not members:
            continue
        logprobs.append(_mean([s.lm.get("lm_mean_logprob", 0.0) for s in members]))
        lengths.append(float(sum(s.n_words for s in members)))
        ttrs.append(_mean([s.stylometry.get("sty_root_ttr", 0.0) for s in members]))
        vectors = [style_vectors[i] for i in para.sentence_indices]
        centroid = [_mean([v[j] for v in vectors]) for j in range(len(doc_style))]
        distances.append(_normalised_distance(centroid, doc_style, style_scales))

    if not logprobs:
        return {}
    threshold = _mean(distances) + _std(distances) if len(distances) > 1 else float("inf")
    return {
        "shift_paragraph_logprob_std": _std(logprobs),
        "shift_paragraph_logprob_range": float(max(logprobs) - min(logprobs)),
        "shift_paragraph_len_cv": _cv(lengths),
        "shift_paragraph_ttr_std": _std(ttrs),
        "shift_paragraph_style_distance_max": float(max(distances)) if distances else 0.0,
        "shift_paragraph_count_with_shift": float(
            sum(1 for d in distances if d > threshold)
        ),
    }
