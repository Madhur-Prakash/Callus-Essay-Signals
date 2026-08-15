"""Tests for the deterministic explanation engine and style-shift layer."""

from __future__ import annotations

import pytest

from app.services.explanation_engine import (
    NOTABLE_STRENGTH,
    ExplanationEngine,
    essay_context_from,
)
from app.services.nlp import segment
from app.services.style_shift import annotate
from app.services.stylometry import measure_sentence

REFERENCE = {
    "document": {
        "scope": "document/train",
        "n_samples": 100,
        "features": {
            "whole_lm_frac_top1": {
                "human": {
                    "mean": 0.30,
                    "std": 0.05,
                    "p5": 0.20,
                    "p10": 0.22,
                    "p25": 0.26,
                    "p50": 0.30,
                    "p75": 0.34,
                    "p90": 0.38,
                    "p95": 0.40,
                }
            },
            "bur_cv_sent_len": {
                "human": {
                    "mean": 0.55,
                    "std": 0.12,
                    "p5": 0.35,
                    "p10": 0.40,
                    "p25": 0.48,
                    "p50": 0.55,
                    "p75": 0.62,
                    "p90": 0.70,
                    "p95": 0.75,
                }
            },
        },
    },
    "sentence": {
        "scope": "sentence/train",
        "n_samples": 1000,
        "features": {
            "lm_frac_top1": {
                "human": {
                    "mean": 0.30,
                    "std": 0.10,
                    "p5": 0.10,
                    "p10": 0.15,
                    "p25": 0.23,
                    "p50": 0.30,
                    "p75": 0.38,
                    "p90": 0.45,
                    "p95": 0.50,
                }
            },
            "lm_perplexity": {
                "human": {
                    "mean": 40.0,
                    "std": 15.0,
                    "p5": 15.0,
                    "p10": 20.0,
                    "p25": 30.0,
                    "p50": 40.0,
                    "p75": 52.0,
                    "p90": 65.0,
                    "p95": 75.0,
                }
            },
        },
    },
}


@pytest.fixture
def engine() -> ExplanationEngine:
    return ExplanationEngine(REFERENCE)


class TestPercentileMapping:
    def test_value_at_the_median_maps_to_the_50th_percentile(self, engine) -> None:  # noqa: ANN001
        assert engine._percentile("document", "whole_lm_frac_top1", 0.30) == pytest.approx(
            50.0, abs=0.1
        )

    def test_values_below_and_above_the_table_clamp(self, engine) -> None:  # noqa: ANN001
        assert engine._percentile("document", "whole_lm_frac_top1", 0.01) == 0.0
        assert engine._percentile("document", "whole_lm_frac_top1", 0.99) == 100.0

    def test_interpolates_between_stored_points(self, engine) -> None:  # noqa: ANN001
        percentile = engine._percentile("document", "whole_lm_frac_top1", 0.32)
        assert 50.0 < percentile < 75.0

    def test_unknown_feature_returns_none_rather_than_guessing(self, engine) -> None:  # noqa: ANN001
        assert engine._percentile("document", "not_a_feature", 1.0) is None


class TestDirectionality:
    def test_high_predictability_reads_as_machine_leaning(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document({"whole_lm_frac_top1": 0.42})
        meter = next(m for m in evidence.meters if m.key == "lm_predictability")
        assert meter.strength > NOTABLE_STRENGTH
        assert meter.level in {"elevated", "high"}

    def test_low_predictability_reads_as_human_leaning(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document({"whole_lm_frac_top1": 0.21})
        meter = next(m for m in evidence.meters if m.key == "lm_predictability")
        assert meter.strength < 0.35
        assert meter.level == "low"

    def test_low_sentence_variation_reads_as_machine_leaning(self, engine) -> None:  # noqa: ANN001
        """Uniformity is machine-leaning, so a LOW coefficient of variation must
        produce a HIGH strength - the inverted direction is easy to get wrong."""
        evidence = engine.explain_document({"bur_cv_sent_len": 0.36})
        meter = next(m for m in evidence.meters if m.key == "sentence_uniformity")
        assert meter.strength > NOTABLE_STRENGTH

    def test_high_sentence_variation_reads_as_human_leaning(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document({"bur_cv_sent_len": 0.74})
        meter = next(m for m in evidence.meters if m.key == "sentence_uniformity")
        assert meter.strength < 0.35

    def test_detail_text_describes_the_value_not_the_verdict(self, engine) -> None:  # noqa: ANN001
        low = engine.explain_document({"bur_cv_sent_len": 0.36})
        high = engine.explain_document({"bur_cv_sent_len": 0.74})
        low_meter = next(m for m in low.meters if m.key == "sentence_uniformity")
        high_meter = next(m for m in high.meters if m.key == "sentence_uniformity")
        assert "cluster tightly" in low_meter.detail
        assert "vary substantially" in high_meter.detail

    def test_value_level_tracks_the_value_not_the_signal(self, engine) -> None:  # noqa: ANN001
        """For an inverted feature a LOW value gives a HIGH signal. The statement
        must not therefore claim the value is high - that is self-contradictory."""
        evidence = engine.explain_document({"bur_cv_sent_len": 0.36})
        meter = next(m for m in evidence.meters if m.key == "sentence_uniformity")
        assert meter.level in {"elevated", "high"}, "signal should be machine-leaning"
        assert meter.value_level in {"below", "well below"}, "value is below the median"

        statement = meter.sentence()
        assert "below the human training median" in statement
        assert "cluster tightly" in statement
        assert "Signal strength:" in statement
        # The old phrasing produced "variation is high ... close to the same length".
        assert "variation is high" not in statement.lower()

    def test_statement_and_detail_never_contradict(self, engine) -> None:  # noqa: ANN001
        for value in (0.36, 0.55, 0.74):
            evidence = engine.explain_document({"bur_cv_sent_len": value})
            meter = next(m for m in evidence.meters if m.key == "sentence_uniformity")
            statement = meter.sentence()
            says_uniform = "cluster tightly" in statement
            says_varied = "vary substantially" in statement
            assert says_uniform != says_varied, "exactly one description must appear"
            if says_uniform:
                assert meter.value_level in {"below", "well below", "close to"}
            else:
                assert meter.value_level in {"above", "well above", "close to"}


class TestMissingReference:
    def test_marks_measurements_as_not_comparable_without_a_reference(self) -> None:
        engine = ExplanationEngine({})
        evidence = engine.explain_document({"whole_lm_frac_top1": 0.42})
        meters = [m for m in evidence.meters if m.key == "lm_predictability"]
        assert meters and meters[0].available is False
        assert "no training reference" in meters[0].reference

    def test_never_claims_a_percentile_it_cannot_compute(self) -> None:
        engine = ExplanationEngine({})
        evidence = engine.explain_document({"whole_lm_frac_top1": 0.42})
        assert evidence.meters, "meters should still be reported, just marked unavailable"
        for meter in evidence.meters:
            if not meter.available:
                assert meter.percentile is None
                assert meter.to_dict()["percentile_vs_human"] is None


class TestStatements:
    def test_reports_concrete_repeated_phrases(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document(
            {"whole_lm_frac_top1": 0.42},
            repeated_phrases=[
                {"phrase": "transformative journey", "count": 3, "length": 2, "sentence_indices": [1]}
            ],
        )
        assert any("transformative journey" in s for s in evidence.statements)
        assert any("3 times" in s for s in evidence.statements)

    def test_reports_repeated_syntactic_templates(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document(
            {},
            repeated_templates=[
                {"template": "PRON VERB DET NOUN", "sentence_count": 4, "sentence_indices": [0]}
            ],
        )
        assert any("PRON VERB DET NOUN" in s for s in evidence.statements)

    def test_reports_the_uniform_sentence_count_from_the_rhythm_series(self, engine) -> None:  # noqa: ANN001
        rhythm = [{"index": i, "words": 20.0} for i in range(8)]
        evidence = engine.explain_document({}, rhythm=rhythm)
        assert any("within 20%" in s for s in evidence.statements)

    def test_names_flagged_paragraphs_one_indexed(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document({}, flagged_paragraphs=[0, 2])
        assert any("paragraphs 1, 3" in s for s in evidence.statements)

    def test_says_so_when_nothing_stood_out(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_document({"whole_lm_frac_top1": 0.30})
        assert any("No individual measurement stood out" in s for s in evidence.statements)

    def test_no_statement_asserts_authorship(self, engine) -> None:  # noqa: ANN001
        """The engine describes measurements. It must never claim who wrote the text."""
        evidence = engine.explain_document(
            {"whole_lm_frac_top1": 0.42, "bur_cv_sent_len": 0.36},
            repeated_phrases=[
                {"phrase": "a testament to", "count": 2, "length": 3, "sentence_indices": [0]}
            ],
        )
        forbidden = ("was written by", "is ai-written", "the author used ai", "definitely")
        joined = " ".join(evidence.statements).lower()
        for phrase in forbidden:
            assert phrase not in joined


class TestSentenceEvidence:
    def test_compares_against_the_essay_baseline(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_sentence(
            {"lm_perplexity": 14.2, "sty_n_words": 28.0, "sty_root_ttr": 3.0},
            essay_context={
                "median_perplexity": 31.8,
                "mean_words": 17.4,
                "median_root_ttr": 3.1,
            },
            score=0.82,
        )
        joined = " ".join(evidence.statements)
        assert "14.2" in joined and "31.8" in joined
        perplexity = next(m for m in evidence.measurements if m["key"] == "lm_perplexity")
        assert "31.8" in perplexity["reference"]

    def test_reports_a_within_essay_z_score_shift(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_sentence(
            {"ctx_z_mean_logprob": 2.4},
            essay_context={},
            score=0.8,
        )
        assert any("standard deviations" in s for s in evidence.statements)

    def test_lists_the_most_predictable_tokens(self, engine) -> None:  # noqa: ANN001
        evidence = engine.explain_sentence(
            {},
            essay_context={},
            score=0.8,
            token_evidence=[
                {"token": "journey", "probability": 0.82, "logprob": -0.2, "rank": 1,
                 "was_model_top_choice": True}
            ],
        )
        assert any("journey" in s for s in evidence.statements)

    def test_includes_the_engine_version_for_traceability(self, engine) -> None:  # noqa: ANN001
        payload = engine.explain_sentence({}, essay_context={}, score=0.5).to_dict()
        assert payload["engine_version"]


class TestEssayContext:
    def test_derives_medians_from_the_document(self) -> None:
        document = segment(
            "I built a robot. "
            "The servo burned out twice that winter because the wiring was wrong. "
            "It failed."
        )
        for sentence in document.sentences:
            measure_sentence(sentence)
        context = essay_context_from(document)
        assert context["n_sentences"] == 3
        assert context["mean_words"] > 0
        assert context["std_words"] > 0


class TestStyleShift:
    def test_a_register_change_produces_a_measurable_shift(self) -> None:
        text = (
            "I built a robot in my garage. It did not work. My dad kept telling me to "
            "read the manual, which I refused to do.\n\n"
            "Furthermore, the endeavour necessitated a comprehensive reconsideration of "
            "my methodological assumptions. Moreover, the experience instilled within me "
            "a profound appreciation for iterative refinement. Consequently, the process "
            "cultivated substantial intellectual humility.\n\n"
            "Anyway. I fixed it in October. It rolled two feet and fell off the table."
        )
        document = segment(text)
        for sentence in document.sentences:
            measure_sentence(sentence)
        shift = annotate(document)

        assert shift["shift_max_style_distance_to_doc"] > 0
        assert shift["shift_mean_pos_js_to_doc"] >= 0
        for sentence in document.sentences:
            assert sentence.context, "every sentence must get a ctx_ block"
            assert -6.0 <= sentence.context["ctx_z_n_words"] <= 6.0

    def test_uniform_text_shows_less_shift_than_mixed_register_text(self) -> None:
        uniform = segment(
            " ".join(["The team completed the assigned task within the allotted time."] * 6)
        )
        mixed = segment(
            "It broke. "
            "Furthermore, the comprehensive reconsideration of methodological assumptions "
            "necessitated substantial intellectual humility and sustained attention. "
            "Twice. "
            "Moreover, the endeavour cultivated a profound appreciation for iteration."
        )
        for document in (uniform, mixed):
            for sentence in document.sentences:
                measure_sentence(sentence)
        assert (
            annotate(uniform)["shift_max_style_distance_to_doc"]
            < annotate(mixed)["shift_max_style_distance_to_doc"]
        )

    def test_single_sentence_document_is_handled(self) -> None:
        document = segment("Only one sentence in this whole document here.")
        for sentence in document.sentences:
            measure_sentence(sentence)
        shift = annotate(document)
        assert shift["shift_max_abs_z_logprob"] == 0.0
