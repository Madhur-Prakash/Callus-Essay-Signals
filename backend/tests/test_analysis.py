"""End-to-end analysis tests through the API, plus persistence and cache behaviour."""

from __future__ import annotations

import pytest

PREFIX = "/api/v1"


@pytest.fixture(scope="module")
def analysis(client, human_essay):  # noqa: ANN001, ANN201
    response = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
    if response.status_code == 503:
        pytest.skip("detector model not trained; run ml.training.train")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.mark.slow
class TestAnalysisResponseShape:
    def test_top_level_fields_are_present(self, analysis) -> None:  # noqa: ANN001
        for key in (
            "analysis_id",
            "status",
            "classification",
            "label",
            "confidence",
            "confidence_score",
            "probabilities",
            "summary",
            "paragraphs",
            "sentences",
            "evidence",
            "rhythm",
            "repetition",
            "model",
            "content_hash",
            "disclaimer",
        ):
            assert key in analysis, f"missing {key}"

    def test_classification_is_one_of_the_four_categories(self, analysis) -> None:  # noqa: ANN001
        assert analysis["classification"] in {
            "human",
            "ai_generated",
            "ai_polished",
            "insufficient_evidence",
        }

    def test_probabilities_form_a_distribution(self, analysis) -> None:  # noqa: ANN001
        total = sum(analysis["probabilities"].values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_disclaimer_states_it_is_not_proof(self, analysis) -> None:  # noqa: ANN001
        assert "not proof of authorship" in analysis["disclaimer"]

    def test_summary_counts_are_internally_consistent(self, analysis) -> None:  # noqa: ANN001
        summary = analysis["summary"]
        assert summary["n_sentences"] == len(analysis["sentences"])
        assert summary["n_paragraphs"] == len(analysis["paragraphs"])
        assert (
            summary["flagged_sentences"]
            + summary["uncertain_sentences"]
            + summary["human_like_sentences"]
            == summary["sentences_scored"]
        )

    def test_model_block_names_the_instrument_and_its_role(self, analysis) -> None:  # noqa: ANN001
        model = analysis["model"]
        assert model["language_model"]
        assert "does not classify" in model["language_model_role"]
        assert model["model_version"]
        assert model["features_version"]


@pytest.mark.slow
class TestSentenceLevelResults:
    def test_every_sentence_has_offsets_and_a_band(self, analysis) -> None:  # noqa: ANN001
        for sentence in analysis["sentences"]:
            assert sentence["end"] > sentence["start"]
            assert sentence["classification"] in {
                "likely_human",
                "uncertain",
                "possibly_ai_assisted",
                "likely_ai_assisted",
                "unavailable",
            }
            if sentence["score"] is not None:
                assert 0.0 <= sentence["score"] <= 1.0

    def test_offsets_slice_the_submitted_text_back_out(self, analysis, human_essay) -> None:  # noqa: ANN001
        from app.core.security import normalise_essay

        normalised = normalise_essay(human_essay)
        for sentence in analysis["sentences"]:
            assert normalised[sentence["start"] : sentence["end"]] == sentence["text"]

    def test_sentence_ids_are_sequential(self, analysis) -> None:  # noqa: ANN001
        ids = [s["sentence_id"] for s in analysis["sentences"]]
        assert ids == list(range(len(ids)))

    def test_evidence_is_attached_to_flagged_and_uncertain_sentences(self, analysis) -> None:  # noqa: ANN001
        for sentence in analysis["sentences"]:
            score = sentence["score"]
            if score is not None and score >= 0.4:
                assert sentence.get("evidence"), (
                    f"sentence {sentence['sentence_id']} scored {score} without evidence"
                )

    def test_every_attached_evidence_block_has_content(self, analysis) -> None:  # noqa: ANN001
        for sentence in analysis["sentences"]:
            evidence = sentence.get("evidence")
            if evidence:
                assert evidence["statements"]
                assert evidence["engine_version"]

    def test_paragraph_rollup_references_real_sentences(self, analysis) -> None:  # noqa: ANN001
        valid = {s["sentence_id"] for s in analysis["sentences"]}
        for paragraph in analysis["paragraphs"]:
            assert set(paragraph["sentence_ids"]) <= valid
            assert set(paragraph["flagged_sentence_ids"]) <= set(paragraph["sentence_ids"])
            if paragraph["human_likeness"] is not None:
                assert 0.0 <= paragraph["human_likeness"] <= 1.0

    def test_every_sentence_belongs_to_exactly_one_paragraph(self, analysis) -> None:  # noqa: ANN001
        seen: list[int] = []
        for paragraph in analysis["paragraphs"]:
            seen.extend(paragraph["sentence_ids"])
        assert sorted(seen) == sorted(s["sentence_id"] for s in analysis["sentences"])


@pytest.mark.slow
class TestEvidenceAndStatistics:
    def test_document_evidence_has_meters_and_statements(self, analysis) -> None:  # noqa: ANN001
        evidence = analysis["evidence"]
        assert evidence["meters"]
        assert evidence["statements"]

    def test_meter_strengths_are_bounded(self, analysis) -> None:  # noqa: ANN001
        for meter in analysis["evidence"]["meters"]:
            assert 0.0 <= meter["strength"] <= 1.0

    def test_rhythm_series_aligns_with_sentences(self, analysis) -> None:  # noqa: ANN001
        assert len(analysis["rhythm"]) == len(analysis["sentences"])
        for point, sentence in zip(analysis["rhythm"], analysis["sentences"], strict=True):
            assert point["index"] == sentence["sentence_id"]
            assert point["words"] == sentence["n_words"]

    def test_statistics_are_real_numbers(self, analysis) -> None:  # noqa: ANN001
        statistics = analysis["summary"]["statistics"]
        assert statistics["perplexity"] > 0
        assert statistics["mean_words_per_sentence"] > 0
        assert 0.0 <= statistics["fraction_top1_tokens"] <= 1.0
        assert statistics["mean_token_logprob"] < 0

    def test_lm_token_count_is_reported(self, analysis) -> None:  # noqa: ANN001
        assert analysis["summary"]["lm_tokens_scored"] > 0
        assert analysis["summary"]["lm_windows"] >= 1


@pytest.mark.slow
class TestPrivacyBehaviour:
    def test_essay_text_is_not_echoed_at_the_top_level(self, analysis) -> None:  # noqa: ANN001
        assert "text" not in analysis

    def test_response_reports_whether_it_was_persisted(self, analysis) -> None:  # noqa: ANN001
        assert isinstance(analysis["persisted"], bool)

    def test_bootstrap_regime_is_surfaced_as_a_warning(self, analysis) -> None:  # noqa: ANN001
        if analysis["model"].get("data_regime") == "bootstrap":
            joined = " ".join(analysis["warnings"]).lower()
            assert "bootstrap" in joined

    def test_essay_text_never_reaches_the_log_file(self, client, human_essay) -> None:  # noqa: ANN001
        """The strongest privacy guarantee in the system, checked against the file."""
        from pathlib import Path

        from app.config import get_settings

        settings = get_settings()
        log_dir = Path(settings.log_path)
        marker = "insufficient sensor separation"
        assert marker in human_essay

        response = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        if response.status_code == 503:
            pytest.skip("model not trained")

        for log_file in log_dir.glob("*.log"):
            content = log_file.read_text(encoding="utf-8", errors="replace")
            assert marker not in content, f"essay text leaked into {log_file.name}"


@pytest.mark.slow
class TestDeterminismAndCaching:
    def test_the_same_essay_yields_the_same_verdict(self, client, human_essay) -> None:  # noqa: ANN001
        first = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        second = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        if first.status_code == 503:
            pytest.skip("model not trained")
        a, b = first.json(), second.json()
        assert a["classification"] == b["classification"]
        assert a["content_hash"] == b["content_hash"]
        assert a["confidence_score"] == pytest.approx(b["confidence_score"], abs=1e-9)

    def test_content_hash_changes_with_the_text(self, client, human_essay) -> None:  # noqa: ANN001
        first = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        second = client.post(
            f"{PREFIX}/analysis", json={"text": human_essay + " One more sentence here."}
        )
        if first.status_code == 503:
            pytest.skip("model not trained")
        assert first.json()["content_hash"] != second.json()["content_hash"]

    def test_whitespace_only_differences_hash_the_same(self, client, human_essay) -> None:  # noqa: ANN001
        """Normalisation happens before hashing, so re-pasting an essay with
        different line wrapping should hit the same cache entry."""
        first = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        second = client.post(
            f"{PREFIX}/analysis", json={"text": human_essay.replace("\n\n", "\n\n\n\n")}
        )
        if first.status_code == 503:
            pytest.skip("model not trained")
        assert first.json()["content_hash"] == second.json()["content_hash"]


@pytest.mark.slow
class TestDirectionalBehaviour:
    def test_machine_register_scores_higher_than_the_human_draft(
        self, client, human_essay, machine_essay
    ) -> None:  # noqa: ANN001
        """A directional check, not an accuracy claim: the machine-register sample
        should not come out *more* human-like than the hand-written draft."""
        human = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        machine = client.post(f"{PREFIX}/analysis", json={"text": machine_essay})
        if human.status_code == 503:
            pytest.skip("model not trained")

        human_probability = human.json()["probabilities"]["human"]
        machine_probability = machine.json()["probabilities"]["human"]
        assert machine_probability <= human_probability + 0.05


@pytest.mark.integration
@pytest.mark.slow
class TestPersistence:
    def test_a_stored_analysis_can_be_fetched_again(self, client, human_essay) -> None:  # noqa: ANN001
        response = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        if response.status_code == 503:
            pytest.skip("model not trained")
        created = response.json()
        if not created["persisted"]:
            pytest.skip("MongoDB unavailable")

        fetched = client.get(f"{PREFIX}/analysis/{created['analysis_id']}")
        assert fetched.status_code == 200
        body = fetched.json()
        assert body["analysis_id"] == created["analysis_id"]
        assert body["classification"] == created["classification"]

    def test_stored_sentences_keep_offsets_but_not_text(self, client, human_essay) -> None:  # noqa: ANN001
        response = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        if response.status_code == 503:
            pytest.skip("model not trained")
        created = response.json()
        if not created["persisted"]:
            pytest.skip("MongoDB unavailable")

        sentences = client.get(f"{PREFIX}/analysis/{created['analysis_id']}/sentences")
        assert sentences.status_code == 200
        rows = sentences.json()["sentences"]
        assert rows
        for row in rows:
            assert row["end"] > row["start"]
            # SAVE_ESSAYS is false in tests, so the text must not have been kept.
            assert row["text"] == ""

    def test_an_analysis_can_be_deleted(self, client, human_essay) -> None:  # noqa: ANN001
        response = client.post(f"{PREFIX}/analysis", json={"text": human_essay})
        if response.status_code == 503:
            pytest.skip("model not trained")
        created = response.json()
        if not created["persisted"]:
            pytest.skip("MongoDB unavailable")

        deleted = client.delete(f"{PREFIX}/analysis/{created['analysis_id']}")
        assert deleted.status_code == 200
        assert deleted.json()["deleted"] is True
        assert client.get(f"{PREFIX}/analysis/{created['analysis_id']}").status_code == 404

    def test_listing_analyses_excludes_essay_text(self, client) -> None:  # noqa: ANN001
        response = client.get(f"{PREFIX}/analysis")
        if response.status_code == 503:
            pytest.skip("MongoDB unavailable")
        body = response.json()
        assert "items" in body
        assert "text" not in response.text.lower() or '"text":' not in response.text


class TestEvaluationEndpoint:
    def test_reports_availability_without_fabricating_metrics(self, client) -> None:  # noqa: ANN001
        body = client.get(f"{PREFIX}/evaluation").json()
        assert "available" in body
        if body["available"]:
            assert body["report"]["overall"]["n_samples"] > 0
            assert body["report"]["interpretation"]
        else:
            assert body["report"] is None
            assert body["message"]
