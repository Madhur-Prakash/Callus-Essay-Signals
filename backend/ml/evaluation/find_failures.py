"""Failure analysis - including the confidently wrong cases.

Being wrong is expected. Being wrong *with high confidence* is the interesting
failure, because it is the one that would cause real harm if this were pointed at
a real applicant. This script finds those cases and writes up each one:

    Actual / Predicted / Confidence
    Why the model likely failed  (derived from the model's own feature contributions)
    Relevant features            (measured value vs the correct class's training range)
    Possible improvement         (keyed to the feature group that drove the error)

It also lists the false positives on human writing (the errors that matter most
in this application) and the missed machine-generated documents.

Usage
-----
    uv run python -m ml.evaluation.find_failures
    uv run python -m ml.evaluation.find_failures --min-confidence 0.5 --top 5
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger, log_event
from app.services.classifier import detector_models
from app.services.feature_extractor import group_of
from ml.dataset_schema import LABELS, dataset_paths, load_samples

logger = get_logger("ml.failures")

REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "failure_report.json"

MIN_CONFIDENT = 0.55
EXCERPT_CHARS = 320

# What to suggest when a given feature group is what pushed the model wrong.
IMPROVEMENT_BY_GROUP: dict[str, str] = {
    "lm": (
        "The language-model probability features drove this error. distilgpt2 is a "
        "small, 2019-vintage model: text whose vocabulary or topic sits outside its "
        "training distribution looks 'surprising' regardless of who wrote it. Try a "
        "larger instrument model (MODEL_NAME=gpt2-medium), and add features that "
        "normalise surprisal by local entropy rather than using raw perplexity."
    ),
    "stylometric": (
        "Surface stylometry drove this error. These features are sensitive to "
        "register rather than authorship - a formal human writer and a machine share "
        "a lot of surface statistics. Adding more human samples in formal registers "
        "would let the model separate 'formal' from 'machine'."
    ),
    "syntactic": (
        "POS/dependency features drove this error. They are computed by a small "
        "spaCy model whose parses degrade on unusual syntax, which is common in both "
        "creative human writing and second-language writing. Consider a transformer "
        "parser, or restrict syntactic features to aggregate distributions that are "
        "robust to individual parse errors."
    ),
    "burstiness": (
        "Sentence-rhythm features drove this error. Burstiness is a genuinely weak "
        "signal for individual documents: plenty of humans write uniformly, and "
        "prompted models can be asked to vary sentence length. It should carry less "
        "weight, and it needs an interaction with document length (short essays have "
        "unstable variance estimates)."
    ),
    "repetition": (
        "Repetition features drove this error. Repeated n-grams are ambiguous: they "
        "appear in machine text drawing on a phrase bank AND in human drafts written "
        "under time pressure. Separating lexical from syntactic-template repetition "
        "more aggressively would help, since only the latter is machine-specific."
    ),
    "structural": (
        "Document-structure features (length, paragraph shape, readability) drove "
        "this error. These are the features most likely to encode a dataset artefact "
        "rather than a real property of machine writing - worth checking whether the "
        "training classes still differ systematically in length."
    ),
    "style_shift": (
        "Within-document style-shift features drove this error. A shift indicates a "
        "register change, and humans change register legitimately - quoting, moving "
        "from narrative to reflection, or writing a stronger conclusion. This group "
        "should inform the evidence panel more than the verdict."
    ),
    "corpus": (
        "Corpus-similarity features drove this error: the document sat closer to the "
        "machine reference centroid than the human one. This is the feature group "
        "most exposed to a narrow training corpus - with only a few dozen "
        "independent human documents, the human centroid is a poor summary of human "
        "writing in general. It is the first thing that should improve when real "
        "essays are added."
    ),
}


def find_failures(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    split: str = "test",
    min_confidence: float = MIN_CONFIDENT,
    top: int = 3,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    artifacts_dir = Path(artifacts_dir or settings.artifacts_path)
    paths = dataset_paths(data_dir)

    bundle = np.load(paths["features"], allow_pickle=False)
    X_doc = bundle["X_doc"].astype(np.float64)
    y_doc = bundle["y_doc"]
    doc_names = [str(n) for n in bundle["doc_feature_names"]]
    doc_splits = np.array([str(s) for s in bundle["doc_splits"]])

    feature_manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    metadata = feature_manifest["document_metadata"]

    if not detector_models.load(artifacts_dir, force=True):
        raise FileNotFoundError(
            f"No trained model in {artifacts_dir}. Run `uv run python -m ml.training.train`."
        )
    model = detector_models.require()
    reference = detector_models.reference_stats.get("document", {}).get("features", {})

    texts = {s.record_id: s.text for s in load_samples(paths["combined"])}
    sources = {s.record_id: s.source for s in load_samples(paths["combined"])}

    mask = doc_splits == split
    X = X_doc[mask]
    y = y_doc[mask]
    rows = [m for m, keep in zip(metadata, mask, strict=False) if keep]

    probabilities = model.predictor.predict_proba(X[:, model.feature_indices])
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)

    wrong = np.where(predictions != y)[0]
    confident_wrong = sorted(
        (i for i in wrong if confidence[i] >= min_confidence),
        key=lambda i: -confidence[i],
    )
    # If the abstention thresholds mean nothing clears `min_confidence`, fall back
    # to the most confident errors available rather than reporting none - the
    # brief asks for three concrete cases, so the honest move is to show the
    # three worst and say what their confidence actually was.
    fallback_used = False
    if len(confident_wrong) < top:
        fallback_used = True
        confident_wrong = sorted(wrong, key=lambda i: -confidence[i])

    cases: list[dict[str, Any]] = []
    for rank, index in enumerate(confident_wrong[: max(top, 3)], start=1):
        row = rows[index]
        actual = LABELS[int(y[index])]
        predicted = LABELS[int(predictions[index])]
        features = {name: float(X[index, j]) for j, name in enumerate(doc_names)}
        contributions = detector_models.document_contributions(features, top_k=10)

        drivers = _group_drivers(contributions)
        relevant = _relevant_features(features, reference, actual, predicted)
        record_id = str(row.get("record_id"))
        include_excerpt = sources.get(record_id) != "ingested_real"

        cases.append(
            {
                "rank": rank,
                "record_id": record_id,
                "actual": actual,
                "predicted": predicted,
                "confidence": round(float(confidence[index]), 4),
                "probabilities": {
                    label: round(float(probabilities[index][j]), 4)
                    for j, label in enumerate(LABELS)
                },
                "metadata": {
                    "topic": row.get("topic"),
                    "model": row.get("model"),
                    "strategy": row.get("strategy"),
                    "length_band": row.get("length_band"),
                    "n_words": row.get("n_words"),
                    "n_sentences": row.get("n_sentences"),
                    "l2_english": row.get("l2_english"),
                    "source": sources.get(record_id),
                },
                "why_the_model_likely_failed": _explain_failure(
                    actual, predicted, drivers, relevant, row
                ),
                "dominant_feature_groups": drivers,
                "relevant_features": relevant,
                "model_contributions": contributions[:6],
                "possible_improvement": [
                    IMPROVEMENT_BY_GROUP[g["group"]]
                    for g in drivers[:2]
                    if g["group"] in IMPROVEMENT_BY_GROUP
                ]
                or [
                    "No single feature group dominated; this failure looks like the "
                    "model integrating many weak signals in the wrong direction, which "
                    "is what a training set this small produces. More independent "
                    "human documents is the fix."
                ],
                "excerpt": (
                    texts.get(record_id, "")[:EXCERPT_CHARS] if include_excerpt else None
                ),
                "excerpt_withheld_reason": (
                    None
                    if include_excerpt
                    else "Operator-supplied real essay; text is not reproduced in reports."
                ),
            }
        )

    human_index = LABELS.index("human")
    false_positives = [
        _brief(rows[i], y[i], predictions[i], confidence[i], probabilities[i])
        for i in wrong
        if y[i] == human_index
    ]
    false_negatives = [
        _brief(rows[i], y[i], predictions[i], confidence[i], probabilities[i])
        for i in wrong
        if y[i] != human_index and predictions[i] == human_index
    ]
    polished_confusion = [
        _brief(rows[i], y[i], predictions[i], confidence[i], probabilities[i])
        for i in wrong
        if LABELS[int(y[i])] == "ai_polished"
    ]

    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "model": {
            "name": model.name,
            "model_version": detector_models.metadata.get("model_version"),
            "calibration": model.calibration_method,
        },
        "data_regime": detector_models.metadata.get("data_regime"),
        "summary": {
            "n_documents": int(mask.sum()),
            "n_errors": int(len(wrong)),
            "error_rate": round(float(len(wrong) / max(1, mask.sum())), 4),
            "n_confidently_wrong": int(
                sum(1 for i in wrong if confidence[i] >= min_confidence)
            ),
            "confidence_threshold": min_confidence,
            "mean_confidence_when_wrong": round(
                float(confidence[wrong].mean()), 4
            )
            if len(wrong)
            else None,
            "mean_confidence_when_right": round(
                float(confidence[predictions == y].mean()), 4
            )
            if (predictions == y).any()
            else None,
            "fallback_used": fallback_used,
            "fallback_note": (
                "Fewer than the requested number of errors cleared the confidence "
                "threshold, so the most confident errors are reported regardless of "
                "whether they crossed it. Each case lists its actual confidence."
                if fallback_used
                else None
            ),
        },
        "confidently_wrong": cases,
        "false_positives_on_human_writing": {
            "count": len(false_positives),
            "note": (
                "These are the errors that would do real damage: human-written essays "
                "the detector called machine-written or machine-polished."
            ),
            "cases": false_positives[:20],
        },
        "missed_machine_writing": {
            "count": len(false_negatives),
            "note": "Machine or machine-edited documents the detector called human.",
            "cases": false_negatives[:20],
        },
        "ai_polished_confusions": {
            "count": len(polished_confusion),
            "note": (
                "The AI_POLISHED class is the hardest by construction: a lightly "
                "edited human essay is mostly human text. Confusion here is expected "
                "and is the honest limit of the method."
            ),
            "cases": polished_confusion[:20],
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event(
        logger,
        "failures.complete",
        errors=len(wrong),
        confidently_wrong=report["summary"]["n_confidently_wrong"],
        cases_written=len(cases),
        report=REPORT_PATH.name,
    )
    return report


def _brief(
    row: dict[str, Any],
    actual: int,
    predicted: int,
    confidence: float,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "actual": LABELS[int(actual)],
        "predicted": LABELS[int(predicted)],
        "confidence": round(float(confidence), 4),
        "topic": row.get("topic"),
        "model": row.get("model"),
        "strategy": row.get("strategy"),
        "l2_english": row.get("l2_english"),
        "n_words": row.get("n_words"),
        "probabilities": {
            label: round(float(probabilities[j]), 4) for j, label in enumerate(LABELS)
        },
    }


def _group_drivers(contributions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-feature contributions up to feature groups."""
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for entry in contributions:
        group = group_of(entry["feature"]) or "other"
        totals[group] = totals.get(group, 0.0) + abs(float(entry["contribution"]))
        counts[group] = counts.get(group, 0) + 1
    total = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])
    return [
        {
            "group": group,
            "share_of_contribution": round(value / total, 4),
            "n_features": counts[group],
        }
        for group, value in ranked
    ]


def _relevant_features(
    features: dict[str, float],
    reference: dict[str, Any],
    actual: str,
    predicted: str,
) -> list[dict[str, Any]]:
    """Features where the document sits outside its true class's normal range.

    This is the concrete answer to "why did it look like the wrong class?" - the
    measured value is compared against the interquartile range of the class it
    actually belongs to.
    """
    findings: list[dict[str, Any]] = []
    for name, entry in reference.items():
        actual_stats = entry.get(actual)
        predicted_stats = entry.get(predicted)
        if not actual_stats or name not in features:
            continue
        value = features[name]
        low, high = actual_stats.get("p25"), actual_stats.get("p75")
        if low is None or high is None:
            continue
        if low <= value <= high:
            continue
        span = max(abs(high - low), 1e-9)
        deviation = (value - high) / span if value > high else (low - value) / span
        if deviation < 0.5:
            continue
        findings.append(
            {
                "feature": name,
                "group": group_of(name),
                "value": round(value, 5),
                "true_class_iqr": [round(float(low), 5), round(float(high), 5)],
                "true_class_median": round(float(actual_stats.get("p50", 0.0)), 5),
                "predicted_class_median": (
                    round(float(predicted_stats.get("p50", 0.0)), 5)
                    if predicted_stats
                    else None
                ),
                "iqr_widths_outside": round(float(deviation), 2),
                "direction": "above" if value > high else "below",
            }
        )
    findings.sort(key=lambda f: -f["iqr_widths_outside"])
    return findings[:8]


def _explain_failure(
    actual: str,
    predicted: str,
    drivers: list[dict[str, Any]],
    relevant: list[dict[str, Any]],
    row: dict[str, Any],
) -> list[str]:
    """Assemble the narrative from measured facts only."""
    lines: list[str] = []

    top_group = drivers[0]["group"] if drivers else None
    if top_group:
        lines.append(
            f"The {top_group} feature group accounted for "
            f"{drivers[0]['share_of_contribution']:.0%} of the model's decision on this "
            f"document."
        )

    for finding in relevant[:3]:
        lines.append(
            f"`{finding['feature']}` measured {finding['value']:.4g}, which is "
            f"{finding['iqr_widths_outside']:.1f} interquartile widths {finding['direction']} "
            f"the normal range for {actual} documents "
            f"({finding['true_class_iqr'][0]:.4g} to {finding['true_class_iqr'][1]:.4g})"
            + (
                f", and closer to the {predicted} median of "
                f"{finding['predicted_class_median']:.4g}."
                if finding.get("predicted_class_median") is not None
                else "."
            )
        )

    if actual == "human" and predicted in {"ai_generated", "ai_polished"}:
        lines.append(
            "This is a false positive on human writing - the failure mode with real "
            "consequences. The measurements that triggered it describe register, not "
            "authorship: a human writing formally, evenly, and without contractions "
            "produces the same numbers as a machine."
        )
        if row.get("l2_english"):
            lines.append(
                "The document is from the simulated second-language English subset. "
                "L2 writing tends toward simpler connectives and more regular sentence "
                "construction, which is exactly what these features read as machine-like. "
                "This is the documented fairness risk of the whole approach, visible in "
                "a single case."
            )
    elif actual == "ai_polished" and predicted == "human":
        lines.append(
            "A lightly edited essay is still mostly human text. When the edit touched "
            "only grammar or one paragraph, most sentences retain the original author's "
            "statistics, and the document-level aggregate is dominated by them."
        )
    elif actual == "ai_polished" and predicted == "ai_generated":
        lines.append(
            "The edit was heavy enough to overwrite the author's style across the whole "
            "document, so it measures like fully generated text. The distinction between "
            "'heavily rewritten' and 'generated' may not be recoverable from text alone."
        )
    elif actual == "ai_generated" and predicted == "human":
        lines.append(
            "The generator produced text with human-like variation - either it was "
            "prompted to vary sentence length and avoid formal connectives, or sampling "
            "at a high temperature introduced the irregularity these features look for."
        )

    if row.get("n_words") and int(row["n_words"]) < 200:
        lines.append(
            f"At {row['n_words']} words the document is short, so every distributional "
            "estimate behind these features (variance, entropy, percentiles) is noisy."
        )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and explain the detector's failures.")
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    parser.add_argument("--min-confidence", type=float, default=MIN_CONFIDENT)
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    report = find_failures(
        split=args.split, min_confidence=args.min_confidence, top=args.top
    )
    summary = report["summary"]
    logger.info(
        f"failure analysis complete | errors={summary['n_errors']}/"
        f"{summary['n_documents']} confidently_wrong={summary['n_confidently_wrong']} "
        f"cases={len(report['confidently_wrong'])}"
    )
    for case in report["confidently_wrong"]:
        logger.info(
            f"confidently wrong | {case['record_id']} actual={case['actual']} "
            f"predicted={case['predicted']} confidence={case['confidence']:.2f}"
        )


if __name__ == "__main__":
    main()
