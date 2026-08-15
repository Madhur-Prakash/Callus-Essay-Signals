"""Metric computation shared by the evaluation scripts.

Everything returns plain JSON-serialisable structures so the same numbers reach
the report file, MongoDB and the research dashboard without being recomputed
(and without a chance of the UI and the report disagreeing).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.services.calibration import expected_calibration_error, reliability_curve
from ml.dataset_schema import LABELS


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray | None = None
) -> dict[str, Any]:
    """Overall + per-class metrics, confusion matrix, ROC-AUC and calibration."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        log_loss,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    n_classes = len(LABELS)
    labels = list(range(n_classes))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    per_class: dict[str, Any] = {}
    for i, name in enumerate(LABELS):
        true_positive = int(matrix[i, i])
        false_negative = int(matrix[i].sum() - true_positive)
        false_positive = int(matrix[:, i].sum() - true_positive)
        true_negative = int(matrix.sum() - true_positive - false_negative - false_positive)
        per_class[name] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            # The two rates that matter most for a detector used on real students.
            "false_positive_rate": round(
                false_positive / (false_positive + true_negative), 4
            )
            if (false_positive + true_negative)
            else 0.0,
            "false_negative_rate": round(
                false_negative / (false_negative + true_positive), 4
            )
            if (false_negative + true_positive)
            else 0.0,
        }

    classes_present = sorted(int(c) for c in np.unique(y_true))
    metrics: dict[str, Any] = {
        "n_samples": int(len(y_true)),
        "n_classes_present": len(classes_present),
        "classes_present": [LABELS[i] for i in classes_present],
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        # `labels` is passed explicitly so macro F1 always averages over the same
        # three classes as the confusion matrix. Without it sklearn infers labels
        # from the data, and a slice containing a single class scores a meaningless
        # 1.0 - which would then be reported as perfect generalisation.
        "macro_f1": round(
            float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4
        ),
        "weighted_f1": round(
            float(
                f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
            ),
            4,
        ),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "matthews_corrcoef": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": list(LABELS),
            "matrix": matrix.tolist(),
            "row_normalised": [
                [round(float(v / row.sum()), 4) if row.sum() else 0.0 for v in row]
                for row in matrix
            ],
        },
    }

    if probabilities is not None and probabilities.size:
        present = np.unique(y_true)
        try:
            if len(present) == n_classes:
                metrics["roc_auc_ovr_macro"] = round(
                    float(
                        roc_auc_score(
                            y_true, probabilities, multi_class="ovr", average="macro"
                        )
                    ),
                    4,
                )
                metrics["roc_auc_ovo_macro"] = round(
                    float(
                        roc_auc_score(
                            y_true, probabilities, multi_class="ovo", average="macro"
                        )
                    ),
                    4,
                )
            else:
                metrics["roc_auc_note"] = (
                    f"Only {len(present)} of {n_classes} classes present in this slice; "
                    "multi-class ROC-AUC is undefined."
                )
        except ValueError as exc:
            metrics["roc_auc_note"] = str(exc)

        # Per-class one-vs-rest AUC is still meaningful where the class exists.
        per_class_auc: dict[str, float | None] = {}
        for i, name in enumerate(LABELS):
            binary = (y_true == i).astype(int)
            if binary.sum() == 0 or binary.sum() == len(binary):
                per_class_auc[name] = None
                continue
            per_class_auc[name] = round(
                float(roc_auc_score(binary, probabilities[:, i])), 4
            )
        metrics["roc_auc_per_class"] = per_class_auc

        try:
            metrics["log_loss"] = round(
                float(log_loss(y_true, probabilities, labels=labels)), 4
            )
        except ValueError:
            metrics["log_loss"] = None

        brier: dict[str, float] = {}
        for i, name in enumerate(LABELS):
            binary = (y_true == i).astype(int)
            if len(np.unique(binary)) == 2:
                brier[name] = round(
                    float(brier_score_loss(binary, probabilities[:, i])), 4
                )
        metrics["brier_score_per_class"] = brier
        metrics["expected_calibration_error"] = round(
            expected_calibration_error(probabilities, y_true), 4
        )
        metrics["reliability_curve"] = reliability_curve(probabilities, y_true)

    return metrics


def roc_curves(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """One-vs-rest ROC curves, thinned for transport to the dashboard."""
    from sklearn.metrics import roc_curve

    curves: dict[str, Any] = {}
    for i, name in enumerate(LABELS):
        binary = (y_true == i).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, probabilities[:, i])
        curves[name] = _thin([{"fpr": float(a), "tpr": float(b)} for a, b in zip(fpr, tpr, strict=False)])
    return curves


def precision_recall_curves(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    curves: dict[str, Any] = {}
    for i, name in enumerate(LABELS):
        binary = (y_true == i).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary, probabilities[:, i])
        curves[name] = {
            "average_precision": round(
                float(average_precision_score(binary, probabilities[:, i])), 4
            ),
            "points": _thin(
                [
                    {"recall": float(r), "precision": float(p)}
                    for p, r in zip(precision, recall, strict=False)
                ]
            ),
        }
    return curves


def _thin(points: list[dict[str, float]], target: int = 60) -> list[dict[str, float]]:
    """Downsample a curve to ~``target`` points, always keeping the endpoints."""
    if len(points) <= target:
        return [{k: round(v, 5) for k, v in p.items()} for p in points]
    step = len(points) / target
    indices = sorted({int(i * step) for i in range(target)} | {0, len(points) - 1})
    return [{k: round(v, 5) for k, v in points[i].items()} for i in indices]


def slice_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    keys: list[str],
    *,
    min_samples: int = 8,
) -> dict[str, Any]:
    """Metrics broken down by an arbitrary grouping key.

    Slices smaller than ``min_samples`` are reported with their size and a
    ``too_small`` flag instead of a metric, because a precision computed over
    three documents is noise dressed as a number.
    """
    output: dict[str, Any] = {}
    for value in sorted(set(keys)):
        mask = np.array([k == value for k in keys])
        n = int(mask.sum())
        if n < min_samples:
            output[value] = {"n_samples": n, "too_small": True}
            continue
        entry = classification_metrics(y_true[mask], y_pred[mask], probabilities[mask])
        present = entry["classes_present"]
        single_class = len(present) < 2

        row: dict[str, Any] = {
            "n_samples": n,
            "classes_present": present,
            "single_class": single_class,
            "accuracy": entry["accuracy"],
            "macro_f1": entry["macro_f1"],
            "balanced_accuracy": entry["balanced_accuracy"],
            "per_class": {
                name: {
                    "recall": stats["recall"],
                    "precision": stats["precision"],
                    "false_positive_rate": stats["false_positive_rate"],
                    "support": stats["support"],
                }
                for name, stats in entry["per_class"].items()
                if stats["support"] > 0
            },
        }
        if single_class:
            # A slice with one class has no meaningful macro F1 (two of the three
            # per-class terms are undefined) and no meaningful accuracy comparison
            # against a multi-class slice. Recall for the present class is the only
            # honest headline, so it is named explicitly.
            only = present[0]
            row["headline_metric"] = "recall"
            row["headline_value"] = entry["per_class"][only]["recall"]
            row["headline_note"] = (
                f"This slice contains only {only} documents, so macro F1 and accuracy "
                f"are not comparable with multi-class slices. The meaningful number is "
                f"recall for {only}: {entry['per_class'][only]['recall']:.3f} over {n} documents."
            )
        else:
            row["headline_metric"] = "macro_f1"
            row["headline_value"] = entry["macro_f1"]
        output[value] = row
    return output


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float]:
    """Wilson score interval for a proportion.

    Reported alongside every headline rate. With a few dozen documents per class,
    a point estimate on its own invites over-reading; the interval makes the
    uncertainty impossible to miss.
    """
    if total == 0:
        return {"point": 0.0, "lower": 0.0, "upper": 0.0, "n": 0}
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    spread = (
        z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2))
    ) / denominator
    return {
        "point": round(float(p), 4),
        "lower": round(float(max(0.0, centre - spread)), 4),
        "upper": round(float(min(1.0, centre + spread)), 4),
        "n": int(total),
    }
