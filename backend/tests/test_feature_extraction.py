"""Tests for segmentation, stylometry, burstiness, repetition and LM scoring."""

from __future__ import annotations

import math

import pytest

from app.services import burstiness, repetition
from app.services.feature_extractor import (
    document_feature_names,
    group_of,
    select_columns,
    sentence_feature_names,
)
from app.services.nlp import find_paragraph_spans, segment
from app.services.stylometry import (
    count_syllables,
    function_word_profile,
    measure_sentence,
    pos_distribution,
    surface_features,
)


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
class TestSegmentation:
    def test_splits_paragraphs_on_blank_lines(self) -> None:
        text = "First para line one.\n\nSecond para here.\n\nThird one."
        spans = find_paragraph_spans(text)
        assert len(spans) == 3
        for start, end in spans:
            assert text[start:end].strip()

    def test_falls_back_to_single_newlines_when_no_blank_lines(self) -> None:
        text = "Line one here.\nLine two here.\nLine three."
        assert len(find_paragraph_spans(text)) == 3

    def test_sentence_offsets_slice_back_to_the_sentence(self) -> None:
        text = "I built a robot. It did not work. I rebuilt it in January."
        document = segment(text)
        assert document.n_sentences == 3
        for sentence in document.sentences:
            assert text[sentence.start : sentence.end] == sentence.text
            assert sentence.text == sentence.text.strip()

    def test_sentences_are_assigned_to_the_right_paragraph(self) -> None:
        text = "Alpha one two. Beta three four.\n\nGamma five six. Delta seven."
        document = segment(text)
        assert document.n_paragraphs == 2
        first = {s.text for s in document.sentences_in(0)}
        assert any("Alpha" in t for t in first)
        assert all("Gamma" not in t for t in first)

    def test_sentences_are_ordered_and_non_overlapping(self, parsed_human) -> None:  # noqa: ANN001
        sentences = parsed_human.document.sentences
        for previous, current in zip(sentences, sentences[1:], strict=False):
            assert previous.end <= current.start
            assert previous.index < current.index

    def test_empty_and_whitespace_input(self) -> None:
        assert segment("").n_sentences == 0
        assert segment("   \n\n  ").n_sentences == 0

    def test_single_sentence_without_terminal_punctuation(self) -> None:
        document = segment("this has no full stop at the end")
        assert document.n_sentences == 1


# --------------------------------------------------------------------------- #
# Stylometry
# --------------------------------------------------------------------------- #
class TestStylometry:
    def test_word_and_punctuation_counts(self) -> None:
        features = surface_features("I ran, quickly; he did not.")
        assert features["sty_n_words"] == 6
        assert features["sty_comma_rate"] > 0
        assert features["sty_semicolon_rate"] > 0

    def test_contractions_are_detected(self) -> None:
        with_contractions = surface_features("I don't think it's working, I'm sure.")
        without = surface_features("I do not think it is working, I am sure.")
        assert with_contractions["sty_contraction_rate"] > 0
        assert without["sty_contraction_rate"] == 0

    def test_llm_register_phrases_are_detected(self) -> None:
        flagged = surface_features("Through this transformative journey I grew.")
        plain = surface_features("I fixed the servo on a Tuesday in March.")
        assert flagged["sty_llm_phrase_rate"] > plain["sty_llm_phrase_rate"]

    def test_transition_words_are_detected_sentence_initially(self) -> None:
        assert surface_features("Moreover, the design failed.")[
            "sty_sentence_initial_transition"
        ] == 1.0
        assert surface_features("The design failed.")[
            "sty_sentence_initial_transition"
        ] == 0.0

    def test_root_ttr_is_less_length_dependent_than_raw_ttr(self) -> None:
        short = surface_features("The cat sat on the mat.")
        long_text = surface_features(" ".join(["The cat sat on the mat."] * 8))
        # Raw TTR collapses with repetition; root TTR stays comparatively stable.
        assert short["sty_ttr"] > long_text["sty_ttr"]
        assert short["sty_root_ttr"] > 0 and long_text["sty_root_ttr"] > 0

    def test_syllable_counting(self) -> None:
        assert count_syllables("cat") == 1
        assert count_syllables("robot") == 2
        assert count_syllables("engineering") >= 3
        assert count_syllables("") == 0

    def test_empty_text_does_not_divide_by_zero(self) -> None:
        features = surface_features("")
        assert all(math.isfinite(v) for v in features.values())
        assert features["sty_n_words"] == 0

    def test_dependency_depth_is_measured_not_pinned(self, parsed_human) -> None:  # noqa: ANN001
        """Regression test: comparing spaCy tokens by identity never terminates the
        walk to the root, which silently pinned every depth at the loop guard."""
        depths = [
            s.syntax["syn_mean_dep_depth"] for s in parsed_human.document.sentences
        ]
        assert any(d > 0 for d in depths)
        assert all(d < 20 for d in depths)
        assert len(set(depths)) > 1

    def test_pos_distribution_sums_to_one_when_available(self, parsed_human) -> None:  # noqa: ANN001
        sentence = parsed_human.document.sentences[0]
        distribution = pos_distribution(sentence.span, sentence.text)
        total = sum(distribution.values())
        assert total == pytest.approx(1.0, abs=1e-6) or total == 0.0

    def test_function_word_profile_has_stable_width(self) -> None:
        a = function_word_profile("The cat sat on the mat.")
        b = function_word_profile("Robotics is difficult work.")
        assert set(a) == set(b)
        assert a["the"] > b["the"]

    def test_measure_sentence_populates_every_block(self) -> None:
        document = segment("I built a small robot in my garage last summer.")
        sentence = document.sentences[0]
        measure_sentence(sentence)
        assert sentence.stylometry and sentence.syntax
        assert sentence.tokens and sentence.pos_distribution


# --------------------------------------------------------------------------- #
# Burstiness
# --------------------------------------------------------------------------- #
class TestBurstiness:
    def test_uniform_text_has_lower_variation_than_varied_text(self) -> None:
        uniform = segment(
            "The first sentence has exactly seven words here. "
            "The second sentence has exactly seven words here. "
            "The third sentence has exactly seven words here. "
            "The fourth sentence has exactly seven words here."
        )
        varied = segment(
            "It failed. "
            "The servo burned out twice that winter because I had wired the ground "
            "return through a breadboard rail that could not carry the current. "
            "Twice. "
            "I rebuilt the whole chassis over a long weekend in January."
        )
        for document in (uniform, varied):
            for sentence in document.sentences:
                measure_sentence(sentence)

        uniform_features = burstiness.extract(uniform)
        varied_features = burstiness.extract(varied)
        assert uniform_features["bur_cv_sent_len"] < varied_features["bur_cv_sent_len"]
        assert uniform_features["bur_std_sent_len"] < varied_features["bur_std_sent_len"]

    def test_burstiness_index_is_bounded(self) -> None:
        assert -1.0 <= burstiness.burstiness_index([5, 5, 5, 5]) <= 1.0
        assert -1.0 <= burstiness.burstiness_index([1, 40, 3, 25]) <= 1.0

    def test_lag1_autocorrelation_is_bounded(self) -> None:
        assert -1.0 <= burstiness.lag1_autocorrelation([1, 2, 3, 4, 5]) <= 1.0
        assert burstiness.lag1_autocorrelation([5, 5, 5, 5]) == 0.0

    def test_length_entropy_is_zero_for_a_single_bin(self) -> None:
        entropy, normalised = burstiness.length_entropy([10, 11, 12])
        assert entropy == 0.0
        assert normalised == 0.0

    def test_handles_a_single_sentence(self) -> None:
        document = segment("One single sentence only here.")
        for sentence in document.sentences:
            measure_sentence(sentence)
        features = burstiness.extract(document)
        assert all(math.isfinite(v) for v in features.values())

    def test_rhythm_series_matches_sentence_count(self, parsed_human) -> None:  # noqa: ANN001
        assert len(parsed_human.rhythm) == parsed_human.document.n_sentences


# --------------------------------------------------------------------------- #
# Repetition
# --------------------------------------------------------------------------- #
class TestRepetition:
    def test_repeated_phrases_are_found_and_counted(self) -> None:
        document = segment(
            "The transformative journey continued. "
            "Later the transformative journey ended. "
            "A transformative journey is a phrase."
        )
        for sentence in document.sentences:
            measure_sentence(sentence)
        spans = repetition.repeated_spans(document)
        phrases = {s["phrase"] for s in spans}
        assert any("transformative journey" in p for p in phrases)

    def test_function_word_only_ngrams_are_not_reported(self) -> None:
        document = segment(
            "It is one of the things. That is one of the items. This is one of the parts."
        )
        for sentence in document.sentences:
            measure_sentence(sentence)
        spans = repetition.repeated_spans(document)
        for span in spans:
            words = str(span["phrase"]).split()
            assert not all(
                w in {"it", "is", "one", "of", "the", "that", "this"} for w in words
            )

    def test_repetition_features_are_higher_for_repetitive_text(self) -> None:
        repetitive = segment(" ".join(["I built a robot in my garage."] * 5))
        varied = segment(
            "I built a robot. Later I soldered a board. Then everything melted. "
            "My sister laughed for a week. We ordered new parts in March."
        )
        for document in (repetitive, varied):
            for sentence in document.sentences:
                measure_sentence(sentence)
        assert (
            repetition.extract(repetitive)["rep_trigram_repeat_ratio"]
            > repetition.extract(varied)["rep_trigram_repeat_ratio"]
        )

    def test_short_text_skips_ngram_features_gracefully(self) -> None:
        document = segment("Too short.")
        for sentence in document.sentences:
            measure_sentence(sentence)
        features = repetition.extract(document)
        assert features["rep_trigram_repeat_ratio"] == 0.0
        assert all(math.isfinite(v) for v in features.values())


# --------------------------------------------------------------------------- #
# Language-model features
# --------------------------------------------------------------------------- #
@pytest.mark.slow
class TestLanguageModelFeatures:
    def test_every_token_is_scored_exactly_once(self, parsed_human) -> None:  # noqa: ANN001
        scores = parsed_human.token_scores
        assert len(scores.tokens) == scores.total_tokens
        indices = [t.index for t in scores.tokens]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)

    def test_probabilities_and_ranks_are_valid(self, parsed_human) -> None:  # noqa: ANN001
        for token in parsed_human.token_scores.tokens:
            assert token.logprob <= 0.0
            assert 0.0 <= token.prob <= 1.0
            assert token.rank >= 1
            assert token.entropy >= 0.0
            assert token.top1_logprob >= token.logprob - 1e-6

    def test_sentences_receive_lm_features(self, parsed_human) -> None:  # noqa: ANN001
        for sentence in parsed_human.document.sentences:
            assert sentence.lm
            assert math.isfinite(sentence.lm["lm_perplexity"])
            assert sentence.lm["lm_perplexity"] > 0

    def test_long_text_uses_multiple_windows(self, pipeline) -> None:  # noqa: ANN001
        long_text = "\n\n".join(
            [
                "I built a robot in my garage the summer before junior year, and it "
                "never worked properly because the sensors were mounted far too close."
            ]
            * 20
        )
        result = pipeline.extract(long_text)
        assert result.token_scores.total_tokens > 512
        assert result.token_scores.n_windows > 1
        assert len(result.token_scores.tokens) == result.token_scores.total_tokens

    def test_machine_register_is_more_predictable_than_the_human_draft(
        self, parsed_human, parsed_machine
    ) -> None:  # noqa: ANN001
        """A directional sanity check on the instrument, not on the classifier."""
        human_top1 = parsed_human.document_features["whole_lm_frac_top1"]
        machine_top1 = parsed_machine.document_features["whole_lm_frac_top1"]
        assert machine_top1 > human_top1


# --------------------------------------------------------------------------- #
# Feature vectors
# --------------------------------------------------------------------------- #
class TestFeatureVectors:
    def test_vectors_have_stable_width_and_no_nans(
        self, parsed_human, parsed_machine
    ) -> None:  # noqa: ANN001
        doc_names = document_feature_names()
        sent_names = sentence_feature_names()
        for result in (parsed_human, parsed_machine):
            assert set(result.document_features) == set(doc_names)
            assert all(math.isfinite(v) for v in result.document_features.values())
            for row in result.sentence_features:
                assert set(row) == set(sent_names)
                assert all(math.isfinite(v) for v in row.values())

    def test_no_duplicate_feature_names(self) -> None:
        for names in (document_feature_names(), sentence_feature_names()):
            assert len(names) == len(set(names))

    def test_every_feature_belongs_to_a_group(self) -> None:
        unassigned = [n for n in document_feature_names() if group_of(n) is None]
        assert unassigned == []

    def test_feature_group_selection_partitions_the_hybrid_set(self) -> None:
        names = list(document_feature_names())
        lm = set(select_columns(names, ("lm",)))
        stylometric = set(select_columns(names, ("stylometric",)))
        assert lm and stylometric
        assert lm.isdisjoint(stylometric)

    def test_aggregate_features_are_traced_to_the_right_group(self) -> None:
        assert group_of("agg_mean_lm_frac_top1") == "lm"
        assert group_of("whole_lm_log_perplexity") == "lm"
        assert group_of("agg_std_sty_n_words") == "stylometric"
        assert group_of("bur_cv_sent_len") == "burstiness"

    def test_extraction_is_deterministic(self, pipeline, human_essay) -> None:  # noqa: ANN001
        first = pipeline.extract(human_essay).document_features
        second = pipeline.extract(human_essay).document_features
        assert first == second

    def test_timings_are_recorded(self, parsed_human) -> None:  # noqa: ANN001
        assert parsed_human.timings["total_features_ms"] > 0
        assert "lm_scoring_ms" in parsed_human.timings
