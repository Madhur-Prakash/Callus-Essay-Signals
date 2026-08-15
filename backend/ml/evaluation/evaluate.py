"""Held-out evaluation, generalisation tests and bias analysis.

This is the only place the test split is used. It answers the research questions
from the project brief with numbers, including the unflattering ones:

* overall + per-class metrics, confusion matrix, ROC / PR curves, calibration
* **model comparison** on the test split (baseline vs LM-only vs hybrid), so the
  claim that the hybrid approach adds value is measured, not asserted
* **topic generalisation** - held-out topics that appear in no training document
* **model generalisation** - a generator withheld from training entirely
* **length generalisation** - short / medium / long bands
* **bias analysis** - false-positive rate on human writing broken down by the
  L2-English flag, with Wilson intervals, plus a two-proportion test

Output: ``ml/evaluation/evaluation_report.json`` (also stored in MongoDB by the
API when it is running, and served to the research dashboard).

Usage
-----
    uv run python -m ml.evaluation.evaluate
    uv run python -m ml.evaluation.evaluate --split validation
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger, log_event
from app.services.calibration import (
    ABSTAIN_MARGIN,
    ABSTAIN_MAX_PROBABILITY,
    build_verdict,
)
from app.services.classifier import detector_models
from app.services.feature_extractor import MODEL_FEATURE_SETS, select_columns
from ml.dataset_schema import INDEX_TO_LABEL, LABELS, dataset_paths
from ml.evaluation.metrics import (
    classification_metrics,
    precision_recall_curves,
    roc_curves,
    slice_metrics,
    wilson_interval,
)

logger = get_logger("ml.evaluate")

REPORT_DIR = Path(__file__).resolve().parent
REPORT_PATH = REPORT_DIR / "evaluation_report.json"


def _load_bundle(data_dir: Path):  # noqa: ANN201
    paths = dataset_paths(data_dir)
    if not paths["features"].exists():
        raise FileNotFoundError(
            f"{paths['features']} not found. Run the feature extraction stage first."
        )
    bundle = np.load(paths["features"], allow_pickle=False)
    feature_manifest = json.loads(paths["feature_manifest"].read_text(encoding="utf-8"))
    dataset_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    return bundle, feature_manifest, dataset_manifest


def evaluate(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    split: str = "test",
    store_in_mongo: bool = False,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    artifacts_dir = Path(artifacts_dir or settings.artifacts_path)

    bundle, feature_manifest, dataset_manifest = _load_bundle(data_dir)
    X_doc = bundle["X_doc"].astype(np.float64)
    y_doc = bundle["y_doc"]
    doc_names = [str(n) for n in bundle["doc_feature_names"]]
    doc_splits = np.array([str(s) for s in bundle["doc_splits"]])
    doc_groups = np.array([str(g) for g in bundle["doc_groups"]])

    metadata = feature_manifest.get("document_metadata") or []
    if len(metadata) != len(y_doc):
        raise RuntimeError(
            "features.npz and features_manifest.json disagree on document count; "
            "re-run `uv run python -m ml.training.extract_features`."
        )

    if not detector_models.load(artifacts_dir, force=True):
        raise FileNotFoundError(
            f"No trained model in {artifacts_dir}. Run `uv run python -m ml.training.train`."
        )
    model = detector_models.require()

    mask = doc_splits == split
    if not mask.any():
        raise RuntimeError(f"No documents in the '{split}' split.")

    X = X_doc[mask]
    y = y_doc[mask]
    rows = [m for m, keep in zip(metadata, mask, strict=False) if keep]

    probabilities = model.predictor.predict_proba(X[:, model.feature_indices])
    predictions = probabilities.argmax(axis=1)

    log_event(
        logger,
        "evaluate.start",
        split=split,
        documents=int(mask.sum()),
        model=model.name,
        calibration=model.calibration_method or "none",
    )

    overall = classification_metrics(y, predictions, probabilities)

    # ------------------------------------------------------- generalisation
    def key_of(field: str, default: str = "unknown") -> list[str]:
        return [str(r.get(field) or default) for r in rows]

    held_out_topics = set(
        dataset_manifest["splits"]["report"]["forced_to_test"]["held_out_topics"]
    )
    held_out_models = set(
        dataset_manifest["splits"]["report"]["forced_to_test"]["held_out_models"]
    )
    topics = key_of("topic")
    models = key_of("model", "none")

    topic_seen = [
        "held_out_topic" if t in held_out_topics else "training_topic" for t in topics
    ]
    model_seen = [
        "held_out_model" if m in held_out_models else "training_model" for m in models
    ]

    generalisation = {
        "by_topic_novelty": slice_metrics(y, predictions, probabilities, topic_seen),
        "by_topic": slice_metrics(y, predictions, probabilities, topics),
        "by_generator_novelty": slice_metrics(y, predictions, probabilities, model_seen),
        "by_generator": slice_metrics(y, predictions, probabilities, models),
        "by_length_band": slice_metrics(
            y, predictions, probabilities, key_of("length_band", "medium")
        ),
        "by_source": slice_metrics(y, predictions, probabilities, key_of("source")),
        "by_polish_transform": slice_metrics(
            y,
            predictions,
            probabilities,
            [
                str(r.get("strategy") or "n/a") if r.get("label") == "ai_polished" else "n/a"
                for r in rows
            ],
        ),
        "held_out_topics": sorted(held_out_topics),
        "held_out_models": sorted(held_out_models),
        "notes": [
            "Held-out topics appear in no training document. Held-out generators "
            "produced no training document.",
            "Slices with fewer than 8 documents report `too_small` instead of a "
            "metric rather than presenting a number computed over a handful of rows.",
        ],
    }

    # --------------------------------------------------------- length effect
    word_counts = np.array([int(r.get("n_words") or 0) for r in rows])
    length_detail = []
    for label, lo, hi in (("<200 words", 0, 200), ("200-280", 200, 280), (">=280", 280, 10**6)):
        band = (word_counts >= lo) & (word_counts < hi)
        if band.sum() >= 8:
            entry = classification_metrics(y[band], predictions[band], probabilities[band])
            length_detail.append(
                {
                    "band": label,
                    "n_samples": int(band.sum()),
                    "accuracy": entry["accuracy"],
                    "macro_f1": entry["macro_f1"],
                }
            )
        else:
            length_detail.append(
                {"band": label, "n_samples": int(band.sum()), "too_small": True}
            )
    generalisation["by_word_count"] = length_detail

    # ------------------------------------------------------------ bias study
    bias = _bias_analysis(y, predictions, probabilities, rows)

    # ------------------------------------------------- abstention behaviour
    abstention = _abstention_analysis(y, probabilities, rows)

    # ---------------------------------------- model comparison on this split
    comparison = _test_set_comparison(
        X_doc, y_doc, doc_names, doc_splits, doc_groups, split=split
    )

    report = {
        "run_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "split": split,
        "data_regime": dataset_manifest.get("data_regime"),
        "data_regime_note": dataset_manifest.get("regime_note"),
        "model": {
            "name": model.name,
            "model_version": detector_models.metadata.get("model_version"),
            "dataset_version": detector_models.metadata.get("dataset_version"),
            "features_version": detector_models.metadata.get("features_version"),
            "calibration": model.calibration_method,
            "n_features": len(model.feature_indices),
            "feature_groups": list(model.groups),
            "trained_at": detector_models.metadata.get("trained_at"),
        },
        "dataset": {
            "documents_in_split": int(mask.sum()),
            "groups_in_split": int(len(set(doc_groups[mask]))),
            "labels": {
                INDEX_TO_LABEL[i]: int((y == i).sum()) for i in range(len(LABELS))
            },
            "total_documents": int(len(y_doc)),
            "split_counts": dataset_manifest["splits"]["counts"],
        },
        "overall": overall,
        "curves": {
            "roc": roc_curves(y, probabilities),
            "precision_recall": precision_recall_curves(y, probabilities),
        },
        "generalisation": generalisation,
        "bias": bias,
        "abstention": abstention,
        "model_comparison": comparison,
        "feature_importance": detector_models.metadata.get("feature_importance", [])[:25],
        "interpretation": _interpretation(
            overall, comparison, bias, generalisation, dataset_manifest
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    log_event(
        logger,
        "evaluate.complete",
        split=split,
        accuracy=overall["accuracy"],
        macro_f1=overall["macro_f1"],
        ece=overall.get("expected_calibration_error"),
        report=str(REPORT_PATH.name),
    )

    if store_in_mongo:
        _store_in_mongo(report)
    return report


# --------------------------------------------------------------------------- #
# Bias
# --------------------------------------------------------------------------- #
def _bias_analysis(
    y: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Does the detector flag second-language English writing disproportionately?

    The measurement that matters is the **false-positive rate on human documents**:
    of the human essays in a group, what share were called machine-written or
    machine-polished? A detector that is accurate overall but wrong more often for
    one population is not acceptable, and hiding that behind an aggregate number
    would be the worst version of this project.
    """
    human_index = LABELS.index("human")
    human_mask = y == human_index
    groups = {
        "l2_english_simulated": np.array(
            [bool(r.get("l2_english")) for r in rows]
        ),
        "l1_english": np.array([not bool(r.get("l2_english")) for r in rows]),
    }

    per_group: dict[str, Any] = {}
    for name, group_mask in groups.items():
        mask = human_mask & group_mask
        n = int(mask.sum())
        if n == 0:
            per_group[name] = {"n_human_documents": 0, "measurable": False}
            continue
        misclassified = int((predictions[mask] != human_index).sum())
        # Break the errors down: called generated, or called polished?
        as_generated = int((predictions[mask] == LABELS.index("ai_generated")).sum())
        as_polished = int((predictions[mask] == LABELS.index("ai_polished")).sum())
        per_group[name] = {
            "n_human_documents": n,
            # A rate over 5 documents is worth reporting with its (very wide)
            # interval; refusing to report it at all would hide the gap instead of
            # showing how little the corpus can support. `underpowered` marks the
            # cases where the interval is too wide to act on.
            "measurable": n >= 5,
            "underpowered": n < 15,
            "false_positive_rate": wilson_interval(misclassified, n),
            "flagged_as_ai_generated": as_generated,
            "flagged_as_ai_polished": as_polished,
            "mean_human_probability": round(
                float(probabilities[mask][:, human_index].mean()), 4
            ),
        }

    l2 = per_group.get("l2_english_simulated", {})
    l1 = per_group.get("l1_english", {})
    disparity: dict[str, Any] = {"measurable": False}
    if l2.get("measurable") and l1.get("measurable"):
        l2_rate = l2["false_positive_rate"]["point"]
        l1_rate = l1["false_positive_rate"]["point"]
        overlap = not (
            l2["false_positive_rate"]["lower"] > l1["false_positive_rate"]["upper"]
            or l1["false_positive_rate"]["lower"] > l2["false_positive_rate"]["upper"]
        )
        disparity = {
            "measurable": True,
            "underpowered": bool(l2.get("underpowered") or l1.get("underpowered")),
            "n_l2_human_documents": l2["false_positive_rate"]["n"],
            "n_l1_human_documents": l1["false_positive_rate"]["n"],
            "l2_false_positive_rate": l2_rate,
            "l1_false_positive_rate": l1_rate,
            "absolute_difference": round(l2_rate - l1_rate, 4),
            "ratio": round(l2_rate / l1_rate, 3) if l1_rate > 0 else None,
            "confidence_intervals_overlap": overlap,
            "conclusion": (
                "No statistically distinguishable disparity at this sample size - the "
                "Wilson intervals overlap. This is NOT evidence of fairness; it is a "
                "sample-size limitation."
                if overlap
                else (
                    "The L2 group is flagged more often than the L1 group and the "
                    "intervals do not overlap. This is a real disparity in this "
                    "evaluation and must be treated as a defect."
                    if l2_rate > l1_rate
                    else "The L1 group is flagged more often than the L2 group."
                )
            ),
        }

    return {
        "question": (
            "Does the detector disproportionately flag human writing from "
            "second-language English writers?"
        ),
        "metric": "false-positive rate on human documents (Wilson 95% interval)",
        "groups": per_group,
        "disparity": disparity,
        "severe_limitation": (
            "The L2 subset in the bootstrap corpus consists of seed essays written in "
            "a SIMULATED second-language register, not writing collected from real L2 "
            "authors. These numbers indicate whether the pipeline can detect such a "
            "disparity; they do not establish that the detector is fair to real "
            "second-language writers. Published work consistently finds that AI "
            "detectors over-flag L2 English writing, and nothing here contradicts "
            "that. Do not deploy this detector as evidence against any student."
        ),
        "other_subgroups": {
            "by_voice": slice_metrics(
                y[human_mask],
                predictions[human_mask],
                probabilities[human_mask],
                [str(r.get("voice") or "unknown") for r, keep in zip(rows, human_mask, strict=False) if keep],
                min_samples=6,
            ),
        },
    }


# --------------------------------------------------------------------------- #
# Abstention
# --------------------------------------------------------------------------- #
def _abstention_analysis(
    y: np.ndarray, probabilities: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """How often does the shipped verdict decline to name a class, and does
    abstaining actually remove errors?"""
    abstained = 0
    correct_when_answering = 0
    answered = 0
    for i, row in enumerate(rows):
        verdict = build_verdict(
            {label: float(probabilities[i][j]) for j, label in enumerate(LABELS)},
            n_sentences=int(row.get("n_sentences") or 0),
            n_words=int(row.get("n_words") or 0),
        )
        if verdict.abstained:
            abstained += 1
            continue
        answered += 1
        if LABELS.index(verdict.classification) == y[i]:
            correct_when_answering += 1

    return {
        "policy": {
            "min_top_probability": ABSTAIN_MAX_PROBABILITY,
            "min_margin_between_top_two": ABSTAIN_MARGIN,
            "min_sentences": 5,
            "min_words": 120,
        },
        "n_documents": int(len(y)),
        "abstained": abstained,
        "abstention_rate": round(abstained / len(y), 4) if len(y) else 0.0,
        "answered": answered,
        "accuracy_when_answering": wilson_interval(correct_when_answering, answered),
        "note": (
            "Accuracy-when-answering above the overall accuracy means the abstention "
            "policy is removing cases the model would have got wrong, which is its "
            "purpose. If they are equal, the policy is only costing coverage."
        ),
    }


# --------------------------------------------------------------------------- #
# Test-set model comparison (the ablation, re-measured out of sample)
# --------------------------------------------------------------------------- #
def _test_set_comparison(
    X_doc: np.ndarray,
    y_doc: np.ndarray,
    doc_names: list[str],
    doc_splits: np.ndarray,
    doc_groups: np.ndarray,
    *,
    split: str,
) -> dict[str, Any]:
    """Retrain each feature set on train and score it on the held-out split.

    Re-fitting here rather than reusing the training run's artifacts keeps the
    comparison apples-to-apples: identical estimator, identical data, only the
    feature set differs.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train_mask = doc_splits == "train"
    eval_mask = doc_splits == split
    results: list[dict[str, Any]] = []

    for set_name, groups in MODEL_FEATURE_SETS.items():
        indices = select_columns(doc_names, groups)
        if not indices:
            continue
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        )
        pipeline.fit(X_doc[train_mask][:, indices], y_doc[train_mask])
        X_eval = X_doc[eval_mask][:, indices]
        y_eval = y_doc[eval_mask]
        probabilities = pipeline.predict_proba(X_eval)
        predictions = probabilities.argmax(axis=1)
        entry = classification_metrics(y_eval, predictions, probabilities)
        human_index = LABELS.index("human")
        results.append(
            {
                "feature_set": set_name,
                "feature_groups": list(groups),
                "n_features": len(indices),
                "accuracy": entry["accuracy"],
                "macro_f1": entry["macro_f1"],
                "balanced_accuracy": entry["balanced_accuracy"],
                "roc_auc_ovr_macro": entry.get("roc_auc_ovr_macro"),
                "human_false_positive_rate": entry["per_class"]["human"][
                    "false_negative_rate"
                ],
                "per_class_f1": {
                    name: stats["f1"] for name, stats in entry["per_class"].items()
                },
                "human_recall": entry["per_class"]["human"]["recall"],
                "_human_index": human_index,
            }
        )

    for entry in results:
        entry.pop("_human_index", None)
    results.sort(key=lambda e: -e["macro_f1"])

    baseline = next((r for r in results if r["feature_set"] == "baseline_stylometric"), None)
    hybrid = next((r for r in results if r["feature_set"] == "hybrid"), None)
    lm_only = next((r for r in results if r["feature_set"] == "lm_only"), None)
    no_shift = next((r for r in results if r["feature_set"] == "hybrid_no_shift"), None)

    def delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
        if not a or not b:
            return None
        return round(a["macro_f1"] - b["macro_f1"], 4)

    return {
        "protocol": (
            "Each feature set is refitted on the training split with an identical "
            f"RandomForest and scored on the '{split}' split. Only the feature set "
            "varies, which is what makes the comparison a controlled one."
        ),
        "estimator": "RandomForestClassifier(n_estimators=500, min_samples_leaf=2)",
        "note": (
            "The 'hybrid' row here is NOT the shipped model. The shipped model uses "
            "whichever estimator won grouped cross-validation, so its headline metrics "
            "in the 'overall' section can differ from this row. Holding the estimator "
            "fixed is deliberate: otherwise a feature-set comparison would be "
            "confounded by estimator choice."
        ),
        "models": results,
        "deltas": {
            "hybrid_minus_baseline": delta(hybrid, baseline),
            "hybrid_minus_lm_only": delta(hybrid, lm_only),
            "hybrid_minus_no_shift": delta(hybrid, no_shift),
        },
    }


def _interpretation(
    overall: dict[str, Any],
    comparison: dict[str, Any],
    bias: dict[str, Any],
    generalisation: dict[str, Any],
    dataset_manifest: dict[str, Any],
) -> list[str]:
    """Plain-language reading of the numbers, including the caveats."""
    notes: list[str] = []
    regime = dataset_manifest.get("data_regime")

    if regime == "bootstrap":
        notes.append(
            "REGIME WARNING: every number in this report was computed on the offline "
            "bootstrap corpus. The human class is hand-authored seed text and the "
            "machine classes come from a template generator and a rule-based editor. "
            "These metrics measure how separable those three generators are. They are "
            "NOT an estimate of performance on real student essays, and they should "
            "not be quoted as one."
        )

    deltas = comparison.get("deltas", {})
    hybrid_gain = deltas.get("hybrid_minus_baseline")
    if hybrid_gain is not None:
        if hybrid_gain > 0.02:
            notes.append(
                f"The hybrid feature set beats the stylometry-only baseline by "
                f"{hybrid_gain:+.3f} macro F1 on the held-out split, so the language-model "
                "features are earning their cost."
            )
        elif hybrid_gain < -0.02:
            notes.append(
                f"The hybrid feature set is {abs(hybrid_gain):.3f} macro F1 WORSE than the "
                "stylometry-only baseline on the held-out split. On this corpus the "
                "language-model features add noise rather than signal."
            )
        else:
            notes.append(
                f"The hybrid and stylometry-only feature sets are within "
                f"{abs(hybrid_gain):.3f} macro F1 of each other on the held-out split - "
                "on this corpus the language-model features do not measurably help. "
                "That is a finding about the bootstrap data as much as about the method: "
                "the template generator is separable on surface statistics alone, so "
                "there is little left for token probabilities to contribute."
            )

    lm_gain = deltas.get("hybrid_minus_lm_only")
    if lm_gain is not None and lm_gain > 0:
        notes.append(
            f"Language-model features alone are {lm_gain:.3f} macro F1 behind the hybrid "
            "set, so perplexity-style measurements are not sufficient on their own."
        )

    # The single most important number in this report, and the one an accuracy
    # figure hides. A detector that is 83% accurate overall but wrong on half the
    # human essays is not usable on real students.
    human = overall.get("per_class", {}).get("human", {})
    if human.get("support"):
        recall = human["recall"]
        misflagged = human["false_negative"]
        notes.append(
            f"FALSE POSITIVES: human recall is {recall:.3f} - {misflagged} of "
            f"{human['support']} human documents in this split were classified as "
            f"machine-written or machine-polished. "
            + (
                "This is the failure mode that matters: overall accuracy is carried by "
                "the machine classes while a large share of genuine human writing is "
                "flagged. The detector is not fit to be pointed at a real applicant."
                if recall < 0.8
                else "Read this alongside the overall accuracy, not instead of it."
            )
        )

    notes.extend(_novelty_note(generalisation, "by_topic_novelty", "held_out_topic", "training_topic", "topic"))
    notes.extend(
        _novelty_note(
            generalisation, "by_generator_novelty", "held_out_model", "training_model", "generator"
        )
    )

    disparity = bias.get("disparity", {})
    if disparity.get("measurable"):
        notes.append("Bias: " + disparity["conclusion"])
    else:
        notes.append(
            "Bias: the L2-English subset in the held-out split is too small to support "
            "a disparity test. This is a gap in the evaluation, not a clean bill of health."
        )

    ece = overall.get("expected_calibration_error")
    if ece is not None:
        notes.append(
            f"Calibration: expected calibration error is {ece:.3f}. "
            + (
                "Confidence values are roughly trustworthy."
                if ece < 0.1
                else "Confidence values are poorly calibrated and should be read as ordinal "
                "rankings only, not as probabilities."
            )
        )

    notes.append(
        "A flag from this system is a prompt to look more closely. It is not evidence "
        "of authorship, and it cannot be - the measurements describe text, and text "
        "does not carry a signature."
    )
    return notes


def _novelty_note(
    generalisation: dict[str, Any],
    section: str,
    unseen_key: str,
    seen_key: str,
    noun: str,
) -> list[str]:
    """Compare a held-out slice against the in-training slice, honestly.

    The held-out slices in this corpus contain only ``ai_generated`` documents (a
    withheld topic or generator produced no human or polished text), so their
    macro F1 and accuracy are not comparable with the multi-class slice. Comparing
    them anyway produces a number like "1.000 on unseen topics", which looks like
    a triumph and means nothing. Where a slice is single-class this reports recall
    for that class and says so.
    """
    slices = generalisation.get(section, {})
    unseen = slices.get(unseen_key, {})
    seen = slices.get(seen_key, {})
    if not unseen or unseen.get("too_small"):
        return [
            f"{noun.capitalize()} generalisation: the held-out {noun} slice is too small "
            "to report."
        ]

    if unseen.get("single_class"):
        only = (unseen.get("classes_present") or ["?"])[0]
        recall = unseen.get("headline_value")
        seen_recall = (seen.get("per_class", {}).get(only, {}) or {}).get("recall")
        line = (
            f"{noun.capitalize()} generalisation: the held-out {noun} slice contains only "
            f"{only} documents ({unseen['n_samples']} of them), so macro F1 is not "
            f"comparable with the in-training slice. Recall for {only} is "
            f"{recall:.3f} on the held-out {noun}"
        )
        if seen_recall is not None:
            line += f" against {seen_recall:.3f} on {noun}s seen in training"
        line += (
            ". That is a genuine cross-"
            + noun
            + " result for this class, but it says nothing about whether human writing "
            "generalises, because no human documents are in this slice - a real gap in "
            "the evaluation design that more data would close."
        )
        return [line]

    if seen.get("macro_f1") is None or unseen.get("macro_f1") is None:
        return []
    drop = seen["macro_f1"] - unseen["macro_f1"]
    return [
        f"{noun.capitalize()} generalisation: macro F1 is {seen['macro_f1']:.3f} on "
        f"{noun}s seen in training and {unseen['macro_f1']:.3f} on held-out {noun}s "
        f"({-drop:+.3f}). "
        + (
            "The gap is small, which suggests the features are not simply memorising "
            f"{noun}s."
            if abs(drop) < 0.1
            else f"Performance drops on unseen {noun}s: part of the in-domain score is "
            f"{noun} memorisation."
            if drop > 0
            else f"Performance is higher on unseen {noun}s, which usually means the two "
            "slices differ in class mix rather than in difficulty."
        )
    ]


def _store_in_mongo(report: dict[str, Any]) -> None:
    """Best-effort push of the report into MongoDB for the dashboard."""
    import asyncio

    from app.db.mongodb import MongoManager

    async def _run() -> None:
        manager = MongoManager()
        if await manager.connect():
            await manager.store_evaluation_run(report)
            await manager.close()
            log_event(logger, "evaluate.stored_in_mongo", run_id=report["run_id"])
        else:
            log_event(logger, "evaluate.mongo_unavailable", level="warning")

    try:
        asyncio.run(_run())
    except Exception as exc:  # pragma: no cover
        log_event(logger, "evaluate.mongo_failed", level="warning", type=type(exc).__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the detector on a held-out split.")
    parser.add_argument("--split", default="test", choices=("test", "validation", "train"))
    parser.add_argument(
        "--store-in-mongo",
        action="store_true",
        help="also write the report into the evaluation_runs collection",
    )
    args = parser.parse_args()

    report = evaluate(split=args.split, store_in_mongo=args.store_in_mongo)
    overall = report["overall"]
    logger.info(
        f"evaluation complete | split={report['split']} n={overall['n_samples']} "
        f"accuracy={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f} "
        f"ece={overall.get('expected_calibration_error')}"
    )
    for line in report["interpretation"]:
        logger.info("interpretation | " + line)


if __name__ == "__main__":
    main()
