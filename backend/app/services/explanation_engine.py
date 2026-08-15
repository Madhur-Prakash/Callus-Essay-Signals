"""Deterministic evidence generation.

The rule is absolute: **no language model is asked to explain anything.** Every
statement in the UI is produced by this module from measured feature values via a
fixed pipeline:

    measured value
        -> percentile within the human training distribution
        -> strength in [0, 1]
        -> a templated sentence with the real numbers substituted

Three kinds of evidence are produced, and they are kept separate because they
support different claims:

**Reference-relative** — "predictability sits at the 94th percentile of the
human essays in our training data". Requires ``reference_stats.json``.

**Within-essay** — "perplexity here is 14.2 against an essay median of 31.8".
Requires nothing but the essay itself, so it still works before training and is
the most defensible kind of evidence: the comparison group is the author.

**Model-derived** — the signed per-feature contributions pulled straight out of
the trained classifier (:meth:`DetectorModels.document_contributions`). This is
the model's own arithmetic, not a narrative about it.

Every statement is phrased as a measurement, never as a conclusion about who
wrote the text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger("app.explain")

EXPLANATION_ENGINE_VERSION = "1.0.0"

# A signal is only reported when it is at least this extreme relative to the
# human reference distribution. 0.70 corresponds to roughly the 70th percentile.
NOTABLE_STRENGTH = 0.70
STRONG_STRENGTH = 0.85


@dataclass(slots=True)
class Meter:
    """One labelled bar in the evidence panel."""

    key: str
    label: str
    strength: float
    """0-1, where 1 means "as machine-like as this measurement gets"."""
    level: str
    """``low`` / ``elevated`` / ``high`` — the word shown next to the bar."""
    value: float
    unit: str
    reference: str
    """What the value is being compared against, in words."""
    detail: str
    available: bool = True
    percentile: float | None = None
    """Percentile of the measured value within the human training distribution."""
    display: str = ""
    """The value formatted for display, using the spec's format string."""
    value_level: str = "typical"
    """Where the *value* sits relative to the human distribution: ``well above`` /
    ``above`` / ``typical`` / ``below`` / ``well below``. Distinct from
    :attr:`level`, which describes how machine-leaning the reading is — for an
    inverted feature like sentence-length variation, a *low* value produces a
    *high* signal, and conflating the two produces self-contradictory prose."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "strength": round(self.strength, 4),
            "level": self.level,
            "value": round(self.value, 4),
            "display": self.display,
            "unit": self.unit,
            "reference": self.reference,
            "detail": self.detail,
            "available": self.available,
            "value_level": self.value_level,
            "percentile_vs_human": (
                round(self.percentile, 1) if self.percentile is not None else None
            ),
        }

    def sentence(self) -> str:
        """One unambiguous statement about this measurement.

        Deliberately separates the three things a reader needs: what was measured,
        where it sits relative to human writing, and how strongly that leans
        machine-ward. Collapsing them into "X is high" is what made the earlier
        phrasing contradict its own detail text.
        """
        return (
            f"{self.label}: {self.display} — {self.value_level} the human training "
            f"median ({self.reference.replace('human training median ', '')}). "
            f"{self.detail} Signal strength: {self.level}."
        )


@dataclass(slots=True)
class Evidence:
    meters: list[Meter] = field(default_factory=list)
    statements: list[str] = field(default_factory=list)
    measurements: list[dict[str, Any]] = field(default_factory=list)
    contributions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "meters": [m.to_dict() for m in self.meters],
            "statements": self.statements,
            "measurements": self.measurements,
            "model_contributions": self.contributions,
            "engine_version": EXPLANATION_ENGINE_VERSION,
        }


# --------------------------------------------------------------------------- #
# Meter definitions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class MeterSpec:
    key: str
    label: str
    feature: str
    higher_is_more_machine: bool
    unit: str
    detail_high: str
    detail_low: str
    fmt: str = "{:.2f}"


DOCUMENT_METERS: tuple[MeterSpec, ...] = (
    MeterSpec(
        key="lm_predictability",
        label="Language-model predictability",
        feature="whole_lm_frac_top1",
        higher_is_more_machine=True,
        unit="fraction of tokens",
        detail_high=(
            "A large share of the words are the single most likely continuation "
            "under our reference language model."
        ),
        detail_low="Word choices are frequently ones the reference model did not rank first.",
        fmt="{:.1%}",
    ),
    MeterSpec(
        key="perplexity",
        label="Wording unpredictability (perplexity)",
        feature="whole_lm_perplexity",
        higher_is_more_machine=False,
        unit="perplexity",
        detail_high="Wording is varied and hard for the reference model to anticipate.",
        detail_low=(
            "Wording is unusually easy for the reference model to anticipate. Low "
            "perplexity alone is not evidence of machine authorship — clear, "
            "conventional prose also scores low."
        ),
        fmt="{:.1f}",
    ),
    MeterSpec(
        key="sentence_uniformity",
        label="Sentence-length uniformity",
        feature="bur_cv_sent_len",
        higher_is_more_machine=False,
        unit="coefficient of variation",
        detail_high="Sentence lengths vary substantially from one sentence to the next.",
        detail_low=(
            "Sentence lengths cluster tightly around the average, with little of the "
            "long/short alternation typical of unedited drafting."
        ),
        fmt="{:.2f}",
    ),
    MeterSpec(
        key="rhythm_variation",
        label="Sentence rhythm variation",
        feature="bur_mean_abs_adjacent_diff",
        higher_is_more_machine=False,
        unit="words between neighbours",
        detail_high="Neighbouring sentences differ markedly in length.",
        detail_low="Neighbouring sentences are close to the same length throughout.",
        fmt="{:.1f}",
    ),
    MeterSpec(
        key="lexical_diversity",
        label="Lexical diversity",
        feature="doc_root_type_token_ratio",
        higher_is_more_machine=False,
        unit="root type-token ratio",
        detail_high="A wide vocabulary relative to the length of the text.",
        detail_low="Vocabulary is narrower than in comparable human samples.",
        fmt="{:.2f}",
    ),
    MeterSpec(
        key="structural_repetition",
        label="Repeated syntactic structures",
        feature="rep_pos_fourgram_repeat_ratio",
        higher_is_more_machine=True,
        unit="repeated 4-gram rate",
        detail_high=(
            "The same grammatical skeleton recurs across sentences more than in "
            "comparable human samples."
        ),
        detail_low="Sentence structures are varied.",
        fmt="{:.1%}",
    ),
    MeterSpec(
        key="connective_density",
        label="Formal connective density",
        feature="agg_mean_sty_transition_word_rate",
        higher_is_more_machine=True,
        unit="per 100 words",
        detail_high=(
            "Explicit connectives (moreover, furthermore, ultimately) appear at an "
            "elevated rate."
        ),
        detail_low="Connective use is unremarkable.",
        fmt="{:.2f}",
    ),
    MeterSpec(
        key="style_shift",
        label="Within-essay style shift",
        feature="shift_max_style_distance_to_doc",
        higher_is_more_machine=True,
        unit="normalised style distance",
        detail_high=(
            "At least one passage departs noticeably from the style of the rest of "
            "the essay. This indicates a register change — editing, quotation, or a "
            "deliberate shift — not authorship."
        ),
        detail_low="Style is consistent across the essay.",
        fmt="{:.2f}",
    ),
)

SENTENCE_METERS: tuple[MeterSpec, ...] = (
    MeterSpec(
        key="lm_predictability",
        label="Language-model predictability",
        feature="lm_frac_top1",
        higher_is_more_machine=True,
        unit="fraction of tokens",
        detail_high="Most words here were the reference model's own first choice.",
        detail_low="Several word choices here surprised the reference model.",
        fmt="{:.1%}",
    ),
    MeterSpec(
        key="perplexity",
        label="Wording unpredictability (perplexity)",
        feature="lm_perplexity",
        higher_is_more_machine=False,
        unit="perplexity",
        detail_high="Wording in this sentence is hard to anticipate.",
        detail_low="Wording in this sentence is easy to anticipate.",
        fmt="{:.1f}",
    ),
    MeterSpec(
        key="lexical_diversity",
        label="Lexical diversity",
        feature="sty_root_ttr",
        higher_is_more_machine=False,
        unit="root type-token ratio",
        detail_high="Varied vocabulary for a sentence of this length.",
        detail_low="Repetitive vocabulary for a sentence of this length.",
        fmt="{:.2f}",
    ),
    MeterSpec(
        key="style_shift",
        label="Difference from the rest of the essay",
        feature="ctx_style_distance_to_doc",
        higher_is_more_machine=True,
        unit="normalised style distance",
        detail_high="This sentence's style sits well away from the essay's own baseline.",
        detail_low="This sentence matches the essay's baseline style.",
        fmt="{:.2f}",
    ),
    MeterSpec(
        key="corpus_similarity",
        label="Similarity to machine reference corpus",
        feature="cor_char_sim_delta_ai_human",
        higher_is_more_machine=True,
        unit="cosine similarity difference",
        detail_high=(
            "Character and syntax patterns are closer to the machine reference "
            "corpus than to the human one."
        ),
        detail_low="Patterns are closer to the human reference corpus.",
        fmt="{:+.3f}",
    ),
    MeterSpec(
        key="formal_register",
        label="Formal / nominalised register",
        feature="sty_nominalization_rate",
        higher_is_more_machine=True,
        unit="per 100 words",
        detail_high="Abstract nominalisations (-tion, -ment, -ity) are dense here.",
        detail_low="Register is concrete rather than abstract.",
        fmt="{:.2f}",
    ),
)


class ExplanationEngine:
    """Turns feature values into evidence. Stateless apart from the reference table."""

    def __init__(self, reference_stats: dict[str, Any] | None = None) -> None:
        self.reference_stats = reference_stats or {}

    @property
    def has_reference(self) -> bool:
        return bool(self.reference_stats.get("document", {}).get("features"))

    # ------------------------------------------------------------ internals
    def _reference_entry(self, scope: str, feature: str) -> dict[str, Any] | None:
        features = (self.reference_stats.get(scope) or {}).get("features") or {}
        entry = features.get(feature)
        if not entry:
            return None
        return entry.get("human") or entry.get("overall")

    def _percentile(self, scope: str, feature: str, value: float) -> float | None:
        """Percentile of ``value`` within the human training distribution.

        Interpolates the stored p5-p95 table. Returns ``None`` when the feature
        has no reference entry, so the caller can fall back to within-essay
        evidence instead of inventing a comparison.
        """
        entry = self._reference_entry(scope, feature)
        if not entry:
            return None
        points = [(p, entry.get(f"p{p}")) for p in (5, 10, 25, 50, 75, 90, 95)]
        points = [(p, v) for p, v in points if v is not None]
        if len(points) < 2:
            return None

        if value <= points[0][1]:
            return 0.0
        if value >= points[-1][1]:
            return 100.0
        for (p_lo, v_lo), (p_hi, v_hi) in zip(points[:-1], points[1:], strict=False):
            if v_lo <= value <= v_hi:
                if abs(v_hi - v_lo) < 1e-12:
                    return float(p_lo)
                fraction = (value - v_lo) / (v_hi - v_lo)
                return float(p_lo + fraction * (p_hi - p_lo))
        return None

    def _meter(
        self, spec: MeterSpec, scope: str, features: dict[str, float]
    ) -> Meter | None:
        if spec.feature not in features:
            return None
        value = float(features[spec.feature])
        percentile = self._percentile(scope, spec.feature, value)
        entry = self._reference_entry(scope, spec.feature)

        display = spec.fmt.format(value)

        if percentile is None:
            # No reference distribution: report the raw measurement and say so,
            # rather than implying a comparison we cannot actually make.
            return Meter(
                key=spec.key,
                label=spec.label,
                strength=0.5,
                level="not comparable",
                value=value,
                unit=spec.unit,
                reference="no training reference available for this measurement",
                detail=(
                    f"Measured {display}. Train the detector to compare this against "
                    "the reference corpora."
                ),
                available=False,
                display=display,
            )

        # `strength` is machine-leaningness: high for a high value when high values
        # are machine-like, high for a low value when low values are machine-like.
        strength = percentile / 100.0 if spec.higher_is_more_machine else 1.0 - percentile / 100.0
        if strength >= STRONG_STRENGTH:
            level = "high"
        elif strength >= NOTABLE_STRENGTH:
            level = "elevated"
        elif strength >= 0.35:
            level = "typical"
        else:
            level = "low"

        # `detail_high` describes what a high *value* means, so it is selected by
        # the percentile, independent of which direction is machine-like.
        detail = spec.detail_high if percentile >= 50 else spec.detail_low

        if percentile >= 85:
            value_level = "well above"
        elif percentile >= 65:
            value_level = "above"
        elif percentile > 35:
            value_level = "close to"
        elif percentile > 15:
            value_level = "below"
        else:
            value_level = "well below"

        median = entry.get("p50") if entry else None
        reference = (
            f"human training median {spec.fmt.format(median)}"
            if median is not None
            else "human training distribution"
        )

        return Meter(
            key=spec.key,
            label=spec.label,
            strength=max(0.0, min(1.0, strength)),
            level=level,
            value=value,
            unit=spec.unit,
            reference=reference,
            detail=detail,
            percentile=percentile,
            display=display,
            value_level=value_level,
        )

    # --------------------------------------------------------- document view
    def explain_document(
        self,
        document_features: dict[str, float],
        *,
        rhythm: list[dict[str, float]] | None = None,
        repeated_phrases: list[dict[str, Any]] | None = None,
        repeated_templates: list[dict[str, Any]] | None = None,
        contributions: list[dict[str, Any]] | None = None,
        flagged_paragraphs: list[int] | None = None,
    ) -> Evidence:
        evidence = Evidence()

        for spec in DOCUMENT_METERS:
            meter = self._meter(spec, "document", document_features)
            if meter is not None:
                evidence.meters.append(meter)

        notable = [m for m in evidence.meters if m.available and m.strength >= NOTABLE_STRENGTH]
        notable.sort(key=lambda m: -m.strength)

        for meter in notable:
            evidence.statements.append(meter.sentence())
            evidence.measurements.append(
                {
                    "name": meter.label,
                    "key": meter.key,
                    "value": round(meter.value, 4),
                    "unit": meter.display,
                    "reference": meter.reference,
                    "percentile_vs_human": (
                        round(meter.percentile, 1) if meter.percentile is not None else None
                    ),
                    "strength": round(meter.strength, 3),
                }
            )

        # --- concrete, checkable observations -------------------------------
        if rhythm:
            lengths = [int(r["words"]) for r in rhythm]
            if len(lengths) >= 4:
                mean_len = sum(lengths) / len(lengths)
                within = sum(1 for n in lengths if abs(n - mean_len) <= 0.2 * max(mean_len, 1))
                if within / len(lengths) >= 0.6:
                    evidence.statements.append(
                        f"{within} of {len(lengths)} sentences fall within 20% of the "
                        f"essay's mean length of {mean_len:.1f} words."
                    )
                spread = max(lengths) - min(lengths)
                evidence.measurements.append(
                    {
                        "name": "Sentence length range",
                        "key": "sentence_length_range",
                        "value": spread,
                        "unit": f"words (from {min(lengths)} to {max(lengths)})",
                        "reference": f"mean {mean_len:.1f} words",
                        "percentile_vs_human": None,
                        "strength": None,
                    }
                )

        for phrase in (repeated_phrases or [])[:3]:
            evidence.statements.append(
                f'The phrase "{phrase["phrase"]}" appears {phrase["count"]} times.'
            )
        for template in (repeated_templates or [])[:2]:
            evidence.statements.append(
                f"The grammatical pattern [{template['template']}] recurs in "
                f"{template['sentence_count']} different sentences."
            )
        if flagged_paragraphs:
            listed = ", ".join(str(i + 1) for i in flagged_paragraphs[:6])
            evidence.statements.append(
                f"Passages differing most from the essay's own baseline style: "
                f"paragraph{'s' if len(flagged_paragraphs) > 1 else ''} {listed}."
            )

        evidence.contributions = contributions or []
        if not evidence.statements:
            evidence.statements.append(
                "No individual measurement stood out from the human reference "
                "distributions. The verdict rests on the combination of many weak "
                "signals rather than on any single one."
            )
        return evidence

    # --------------------------------------------------------- sentence view
    def explain_sentence(
        self,
        sentence_features: dict[str, float],
        *,
        essay_context: dict[str, float],
        score: float,
        contributions: list[dict[str, Any]] | None = None,
        token_evidence: list[dict[str, Any]] | None = None,
    ) -> Evidence:
        """Evidence for one sentence.

        ``essay_context`` carries the essay's own medians (see
        :func:`essay_context_from`), which power the within-essay comparisons —
        the most defensible evidence available, because the comparison group is
        the author themself.
        """
        evidence = Evidence()

        for spec in SENTENCE_METERS:
            meter = self._meter(spec, "sentence", sentence_features)
            if meter is not None:
                evidence.meters.append(meter)

        notable = sorted(
            (m for m in evidence.meters if m.available and m.strength >= NOTABLE_STRENGTH),
            key=lambda m: -m.strength,
        )
        for meter in notable[:4]:
            evidence.statements.append(meter.sentence())

        # --- within-essay comparisons --------------------------------------
        comparisons = (
            (
                "Perplexity",
                "lm_perplexity",
                "median_perplexity",
                "{:.1f}",
                "lower than",
                "higher than",
            ),
            (
                "Sentence length",
                "sty_n_words",
                "mean_words",
                "{:.0f}",
                "shorter than",
                "longer than",
            ),
            (
                "Lexical diversity",
                "sty_root_ttr",
                "median_root_ttr",
                "{:.2f}",
                "below",
                "above",
            ),
        )
        for label, feature, context_key, fmt, low_word, high_word in comparisons:
            if feature not in sentence_features or context_key not in essay_context:
                continue
            value = float(sentence_features[feature])
            reference_value = float(essay_context[context_key])
            evidence.measurements.append(
                {
                    "name": label,
                    "key": feature,
                    "value": round(value, 4),
                    "unit": fmt.format(value),
                    "reference": f"essay {'median' if 'median' in context_key else 'average'} "
                    f"{fmt.format(reference_value)}",
                    "percentile_vs_human": self._percentile("sentence", feature, value),
                    "strength": None,
                }
            )
            if reference_value > 1e-9:
                ratio = value / reference_value
                if ratio <= 0.6 or ratio >= 1.6:
                    word = low_word if ratio < 1 else high_word
                    evidence.statements.append(
                        f"{label} is {fmt.format(value)}, substantially {word} the "
                        f"essay's own {fmt.format(reference_value)}."
                    )

        z_logprob = sentence_features.get("ctx_z_mean_logprob")
        if z_logprob is not None and abs(z_logprob) >= 1.5:
            direction = "more predictable" if z_logprob > 0 else "less predictable"
            evidence.statements.append(
                f"Token predictability here is {abs(z_logprob):.1f} standard deviations "
                f"{direction} than the rest of this essay."
            )

        if token_evidence:
            words = ", ".join(f'"{t["token"]}" ({t["probability"]:.0%})' for t in token_evidence[:3])
            evidence.statements.append(
                f"Most predictable words in this sentence: {words}."
            )

        evidence.contributions = contributions or []
        if not evidence.statements:
            evidence.statements.append(
                "Nothing about this sentence stands out from the essay's own baseline "
                "or from the human reference distributions."
            )
        return evidence


def essay_context_from(document, rhythm: list[dict[str, float]] | None = None) -> dict[str, float]:  # noqa: ANN001
    """Essay-level reference values used by the within-essay comparisons."""

    def median(values: list[float]) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        mid = len(ordered) // 2
        return (
            float(ordered[mid])
            if len(ordered) % 2
            else float((ordered[mid - 1] + ordered[mid]) / 2)
        )

    sentences = document.sentences
    lengths = [float(s.n_words) for s in sentences]
    perplexities = [float(s.lm.get("lm_perplexity", 0.0)) for s in sentences if s.lm]
    ttrs = [float(s.stylometry.get("sty_root_ttr", 0.0)) for s in sentences]
    logprobs = [float(s.lm.get("lm_mean_logprob", 0.0)) for s in sentences if s.lm]

    return {
        "mean_words": sum(lengths) / len(lengths) if lengths else 0.0,
        "median_words": median(lengths),
        "median_perplexity": median(perplexities),
        "median_root_ttr": median(ttrs),
        "median_logprob": median(logprobs),
        "std_words": _std(lengths),
        "n_sentences": float(len(sentences)),
    }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
