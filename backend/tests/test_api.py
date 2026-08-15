"""API contract tests: validation, error envelopes, health, model info."""

from __future__ import annotations

import pytest

from app.core.exceptions import (
    EmptyEssayError,
    EssayTooLongError,
    EssayTooShortError,
)
from app.core.security import normalise_essay, validate_essay

PREFIX = "/api/v1"


class TestInputValidation:
    def test_normalisation_removes_zero_width_and_control_characters(self) -> None:
        cleaned = normalise_essay("I built​ a robot here.")
        assert "​" not in cleaned
        assert "" not in cleaned
        assert "I built a robot here." == cleaned

    def test_normalisation_collapses_paragraph_gaps_but_keeps_breaks(self) -> None:
        cleaned = normalise_essay("One.\n\n\n\n\nTwo.")
        assert cleaned == "One.\n\nTwo."

    def test_normalisation_converts_crlf(self) -> None:
        assert "\r" not in normalise_essay("One.\r\n\r\nTwo.")

    def test_normalisation_preserves_the_authors_punctuation(self) -> None:
        """The detector measures style, so normalisation must not 'fix' writing."""
        original = "I don't know -- maybe; perhaps... it worked!"
        cleaned = normalise_essay(original)
        assert "don't" in cleaned
        assert ";" in cleaned and "..." in cleaned and "!" in cleaned

    def test_empty_essay_is_rejected(self, settings) -> None:  # noqa: ANN001
        with pytest.raises(EmptyEssayError):
            validate_essay("", settings)
        with pytest.raises(EmptyEssayError):
            validate_essay("   \n  ", settings)
        with pytest.raises(EmptyEssayError):
            validate_essay(None, settings)

    def test_short_essay_is_rejected_with_the_limit_in_the_message(self, settings) -> None:  # noqa: ANN001
        with pytest.raises(EssayTooShortError) as caught:
            validate_essay("Too short.", settings)
        assert str(settings.min_essay_chars) in str(caught.value).replace(",", "")

    def test_long_essay_is_rejected(self, settings) -> None:  # noqa: ANN001
        with pytest.raises(EssayTooLongError):
            validate_essay("word " * (settings.max_essay_chars), settings)

    def test_valid_essay_passes_through_normalised(self, settings, human_essay) -> None:  # noqa: ANN001
        cleaned = validate_essay(human_essay, settings)
        assert cleaned == normalise_essay(human_essay)


class TestHealth:
    def test_health_reports_every_component(self, client) -> None:  # noqa: ANN001
        response = client.get(f"{PREFIX}/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"ok", "degraded", "unavailable"}
        names = {c["name"] for c in body["components"]}
        for expected in {
            "mongodb",
            "redis",
            "kafka",
            "language_model",
            "spacy",
            "detector_model",
            "corpus_reference",
        }:
            assert expected in names

    def test_disabled_components_are_reported_as_disabled_not_broken(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/health").json()
        redis = next(c for c in body["components"] if c["name"] == "redis")
        assert redis["enabled"] is False
        assert "disabled by configuration" in (redis["detail"] or "")

    def test_liveness_and_readiness(self, client) -> None:  # noqa: ANN001
        assert client.get(f"{PREFIX}/health/live").json() == {"status": "alive"}
        ready = client.get(f"{PREFIX}/health/ready")
        assert ready.status_code in {200, 503}
        assert "ready" in ready.json()

    def test_response_time_header_is_present(self, client) -> None:  # noqa: ANN001
        response = client.get(f"{PREFIX}/health")
        assert "X-Response-Time-Ms" in response.headers


class TestModelInfo:
    def test_reports_versions_and_methodology(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/model/info").json()
        assert "detector_version" in body
        assert body["methodology"]["pipeline"]
        assert body["methodology"]["limitations"]

    def test_states_that_the_language_model_does_not_classify(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/model/info").json()
        description = body["methodology"]["what_the_language_model_does"].lower()
        assert "instrument" in description
        assert "never asked to judge authorship" in description
        assert "no hosted chat model" in description
        decision = body["methodology"]["what_makes_the_decision"].lower()
        assert "classifier" in decision

    def test_limitations_mention_the_second_language_risk(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/model/info").json()
        joined = " ".join(body["methodology"]["limitations"]).lower()
        assert "english" in joined


class TestPrivacyEndpoint:
    def test_reports_what_is_stored_and_never_logged(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/essays/privacy").json()
        assert body["save_essays_default"] is False
        assert "the essay text" in body["what_is_never_stored"]
        assert any("essay text" in item for item in body["what_is_never_logged"])
        assert body["deletion_endpoint"].startswith("DELETE")


class TestErrorEnvelope:
    def test_empty_text_returns_the_specific_empty_code(self, client) -> None:  # noqa: ANN001
        """Whitespace-only input must reach the domain validator, so the caller gets
        `essay_empty` (which the UI renders as "please paste an essay") rather than a
        generic schema `validation_error`."""
        for blank in ("", "   ", "\n\n\t "):
            response = client.post(f"{PREFIX}/analysis", json={"text": blank})
            assert response.status_code == 422, blank
            assert response.json()["error"]["code"] == "essay_empty", blank

    def test_short_text_returns_essay_too_short(self, client) -> None:  # noqa: ANN001
        response = client.post(f"{PREFIX}/analysis", json={"text": "Way too short."})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "essay_too_short"

    def test_oversized_text_is_rejected(self, client, settings) -> None:  # noqa: ANN001
        response = client.post(
            f"{PREFIX}/analysis", json={"text": "word " * (settings.max_essay_chars // 2)}
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] in {"essay_too_long", "request_too_large"}

    def test_missing_field_returns_a_validation_error(self, client) -> None:  # noqa: ANN001
        response = client.post(f"{PREFIX}/analysis", json={})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_unknown_analysis_id_returns_404_or_503(self, client) -> None:  # noqa: ANN001
        response = client.get(f"{PREFIX}/analysis/does-not-exist-000")
        assert response.status_code in {404, 503}
        assert response.json()["error"]["code"] in {
            "analysis_not_found",
            "persistence_unavailable",
        }

    def test_no_python_traceback_leaks_in_any_error_body(self, client) -> None:  # noqa: ANN001
        for payload in ({"text": ""}, {"text": "short"}, {}):
            response = client.post(f"{PREFIX}/analysis", json=payload)
            raw = response.text.lower()
            for leak in ("traceback", 'file "', "line ", "app/services"):
                assert leak not in raw, f"error body leaked internals: {raw[:200]}"

    def test_unknown_route_returns_the_error_envelope(self, client) -> None:  # noqa: ANN001
        response = client.get(f"{PREFIX}/not-a-real-route")
        assert response.status_code == 404
        assert "error" in response.json()


class TestOpenApi:
    def test_schema_is_generated_and_documents_the_analysis_route(self, client) -> None:  # noqa: ANN001
        schema = client.get("/openapi.json").json()
        assert f"{PREFIX}/analysis" in schema["paths"]
        assert "AnalysisRequest" in schema["components"]["schemas"]
        assert "AnalysisResponse" in schema["components"]["schemas"]

    def test_root_states_the_no_wrapper_property(self, client) -> None:  # noqa: ANN001
        body = client.get("/").json()
        assert "own trained classifier" in body["note"]
