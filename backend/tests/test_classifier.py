"""Tests for the classifier, calibration and confidence banding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.services.calibration import (
    ABSTAIN_MARGIN,
    ABSTAIN_MAX_PROBABILITY,
    build_verdict,
    confidence_band,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
    sentence_band,
)


class TestConfidenceBanding:
    def test_bands_are_ordered(self) -> None:
        assert confidence_band(0.95) == "high"
        assert confidence_band(0.72) == "moderate"
        assert confidence_band(0.5) == "low"
        assert confidence_band(0.2) == "very low"

    def test_sentence_band_thresholds(self) -> None:
        assert sentence_band(0.9, n_words=20, n_lm_tokens=25)[0] == "likely_ai_assisted"
        assert sentence_band(0.65, n_words=20, n_lm_tokens=25)[0] == "possibly_ai_assisted"
        assert sentence_band(0.5, n_words=20, n_lm_tokens=25)[0] == "uncertain"
        assert sentence_band(0.1, n_words=20, n_lm_tokens=25)[0] == "likely_human"

    def test_very_short_sentences_are_forced_to_uncertain(self) -> None:
        """A four-word sentence gives the language model three predictions, which
        cannot support a claim in either direction."""
        classification, confidence = sentence_band(0.97, n_words=3, n_lm_tokens=4)
        assert classification == "uncertain"
        assert confidence == "very low"

    def test_short_sentence_rule_applies_to_low_scores_too(self) -> None:
        assert sentence_band(0.02, n_words=2, n_lm_tokens=2)[0] == "uncertain"


class TestVerdictConstruction:
    def _probs(self, human: float, generated: float, polished: float) -> dict[str, float]:
        return {"human": human, "ai_generated": generated, "ai_polished": polished}

    def test_names_the_leading_class_when_it_is_clear(self) -> None:
        verdict = build_verdict(
            self._probs(0.05, 0.05, 0.90), n_sentences=12, n_words=400
        )
        assert verdict.classification == "ai_polished"
        assert verdict.confidence == "high"
        assert verdict.abstained is False

    def test_abstains_when_no_class_is_ahead(self) -> None:
        verdict = build_verdict(
            self._probs(0.34, 0.33, 0.33), n_sentences=12, n_words=400
        )
        assert verdict.classification == "insufficient_evidence"
        assert verdict.abstained is True
        assert "threshold" in (verdict.abstain_reason or "")

    def test_abstains_when_the_top_two_are_close(self) -> None:
        verdict = build_verdict(
            self._probs(0.48, 0.46, 0.06), n_sentences=12, n_words=400
        )
        assert verdict.abstained is True
        assert "within" in (verdict.abstain_reason or "")

    def test_abstains_on_a_document_too_short_to_measure(self) -> None:
        verdict = build_verdict(
            self._probs(0.02, 0.02, 0.96), n_sentences=2, n_words=40
        )
        assert verdict.classification == "insufficient_evidence"
        assert "words" in (verdict.abstain_reason or "")

    def test_abstains_on_empty_probabilities(self) -> None:
        verdict = build_verdict({}, n_sentences=10, n_words=300)
        assert verdict.abstained is True

    def test_thresholds_are_the_documented_ones(self) -> None:
        just_under = build_verdict(
            self._probs(ABSTAIN_MAX_PROBABILITY - 0.01, 0.3, 0.26),
            n_sentences=12,
            n_words=400,
        )
        assert just_under.abstained is True
        assert ABSTAIN_MARGIN == pytest.approx(0.10)

    def test_serialises_without_losing_fields(self) -> None:
        payload = build_verdict(
            self._probs(0.05, 0.05, 0.90), n_sentences=12, n_words=400
        ).to_dict()
        for key in (
            "classification",
            "label",
            "description",
            "confidence",
            "confidence_score",
            "probabilities",
            "margin",
            "abstained",
        ):
            assert key in payload


class TestCalibration:
    def _fitted_model(self, n: int = 240):  # noqa: ANN202
        from sklearn.ensemble import RandomForestClassifier

        rng = np.random.default_rng(0)
        X = rng.normal(size=(n, 12))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        y = np.where(rng.random(n) < 0.3, 2, y)  # make it three-class
        model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)
        return model, rng

    def test_platt_scaling_fits_and_produces_valid_probabilities(self) -> None:
        model, rng = self._fitted_model()
        X_cal = rng.normal(size=(90, 12))
        y_cal = rng.integers(0, 3, size=90)
        calibrated, method = fit_calibrator(model, X_cal, y_cal, method="sigmoid")
        assert method == "sigmoid"
        probabilities = calibrated.predict_proba(X_cal)
        assert probabilities.shape == (90, 3)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_skips_calibration_when_there_is_not_enough_data(self) -> None:
        model, rng = self._fitted_model()
        X_cal = rng.normal(size=(8, 12))
        y_cal = rng.integers(0, 3, size=8)
        _, method = fit_calibrator(model, X_cal, y_cal)
        assert method == "none"

    def test_skips_calibration_when_only_one_class_is_present(self) -> None:
        model, rng = self._fitted_model()
        X_cal = rng.normal(size=(60, 12))
        y_cal = np.zeros(60, dtype=int)
        _, method = fit_calibrator(model, X_cal, y_cal)
        assert method == "none"

    def test_isotonic_downgrades_to_sigmoid_on_a_tiny_class(self) -> None:
        model, rng = self._fitted_model()
        X_cal = rng.normal(size=(60, 12))
        y_cal = np.array([0] * 28 + [1] * 28 + [2] * 4)
        _, method = fit_calibrator(model, X_cal, y_cal, method="isotonic")
        assert method == "sigmoid"

    def test_explicit_none_is_respected(self) -> None:
        model, rng = self._fitted_model()
        X_cal = rng.normal(size=(60, 12))
        y_cal = rng.integers(0, 3, size=60)
        estimator, method = fit_calibrator(model, X_cal, y_cal, method="none")
        assert method == "none"
        assert estimator is model


class TestCalibrationMetrics:
    def test_perfectly_calibrated_predictions_have_low_ece(self) -> None:
        probabilities = np.array([[0.9, 0.05, 0.05]] * 10)
        y_true = np.array([0] * 9 + [1])
        assert expected_calibration_error(probabilities, y_true) < 0.05

    def test_overconfident_predictions_have_high_ece(self) -> None:
        probabilities = np.array([[0.99, 0.005, 0.005]] * 10)
        y_true = np.array([0] * 5 + [1] * 5)
        assert expected_calibration_error(probabilities, y_true) > 0.4

    def test_reliability_curve_bins_cover_the_samples(self) -> None:
        rng = np.random.default_rng(1)
        raw = rng.random((100, 3))
        probabilities = raw / raw.sum(axis=1, keepdims=True)
        y_true = rng.integers(0, 3, size=100)
        curve = reliability_curve(probabilities, y_true)
        assert sum(point["count"] for point in curve) == 100
        for point in curve:
            assert 0.0 <= point["observed_accuracy"] <= 1.0

    def test_empty_input_is_handled(self) -> None:
        assert expected_calibration_error(np.array([]), np.array([])) == 0.0
        assert reliability_curve(np.array([]), np.array([])) == []


@pytest.mark.slow
class TestTrainedModel:
    def test_document_prediction_is_a_valid_distribution(
        self, trained_models, parsed_human
    ) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present; run ml.training.train")
        probabilities = trained_models.predict_document(parsed_human.document_features)
        assert set(probabilities) == {"human", "ai_generated", "ai_polished"}
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)
        assert all(0.0 <= v <= 1.0 for v in probabilities.values())

    def test_sentence_scores_are_bounded_and_one_per_sentence(
        self, trained_models, parsed_human
    ) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present")
        scores = trained_models.predict_sentences(parsed_human.sentence_features)
        assert len(scores) == len(parsed_human.sentence_features)
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_missing_features_default_to_zero_rather_than_raising(
        self, trained_models
    ) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present")
        probabilities = trained_models.predict_document({})
        assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-5)

    def test_non_finite_feature_values_are_sanitised(
        self, trained_models, parsed_human
    ) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present")
        poisoned = dict(parsed_human.document_features)
        first_key = next(iter(poisoned))
        poisoned[first_key] = float("nan")
        probabilities = trained_models.predict_document(poisoned)
        assert all(math.isfinite(v) for v in probabilities.values())

    def test_contributions_are_signed_and_ranked(
        self, trained_models, parsed_machine
    ) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present")
        contributions = trained_models.document_contributions(
            parsed_machine.document_features, top_k=6
        )
        if not contributions:
            pytest.skip("model does not expose weights")
        magnitudes = [abs(c["contribution"]) for c in contributions]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(c["direction"] in {"machine-like", "human-like"} for c in contributions)

    def test_metadata_records_reproducibility_fields(self, trained_models) -> None:  # noqa: ANN001
        if trained_models is None:
            pytest.skip("model artifacts not present")
        metadata = trained_models.metadata
        for key in (
            "model_version",
            "dataset_version",
            "features_version",
            "trained_at",
            "training",
        ):
            assert key in metadata, f"metadata is missing {key}"
