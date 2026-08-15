"""Loading the evaluation and failure reports for the research dashboard.

Reports are read from disk (written by the ML scripts) with MongoDB as an
optional second source. Disk is authoritative because the ML pipeline can be run
without any database at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import BACKEND_ROOT
from app.core.logging import get_logger, log_event

logger = get_logger("app.evaluation")

EVALUATION_REPORT = BACKEND_ROOT / "ml" / "evaluation" / "evaluation_report.json"
FAILURE_REPORT = BACKEND_ROOT / "ml" / "evaluation" / "failure_report.json"
DATASET_MANIFEST = BACKEND_ROOT / "data" / "processed" / "manifest.json"
MODEL_COMPARISON = BACKEND_ROOT / "ml" / "artifacts" / "model_comparison.json"


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log_event(logger, "evaluation.bad_json", level="warning", file=path.name)
        return None


def load_evaluation_bundle() -> dict[str, Any]:
    """Everything the research dashboard needs, in one payload."""
    report = _read(EVALUATION_REPORT)
    failures = _read(FAILURE_REPORT)
    dataset = _read(DATASET_MANIFEST)
    comparison = _read(MODEL_COMPARISON)

    if dataset:
        # The manifest embeds up to 50 near-duplicate findings and full topic
        # tables; the dashboard only needs the summary, and trimming keeps the
        # response small.
        dataset = {
            k: v
            for k, v in dataset.items()
            if k
            in {
                "dataset_version",
                "created_at",
                "data_regime",
                "regime_note",
                "totals",
                "labels",
                "splits",
                "sources",
                "models",
                "strategies",
                "topics",
                "length_bands",
                "l2_english",
                "licenses",
                "preprocessing",
                "known_limitations",
            }
        }
        dataset["leakage_controls"] = {
            k: v
            for k, v in (_read(DATASET_MANIFEST) or {}).get("leakage_controls", {}).items()
            if k != "near_duplicate_findings"
        }

    if report is not None and comparison is not None:
        report["training_time_comparison"] = comparison

    available = report is not None
    return {
        "available": available,
        "report": report,
        "failures": failures,
        "dataset": dataset,
        "message": (
            None
            if available
            else (
                "No evaluation report found. Run "
                "`uv run python -m ml.evaluation.evaluate` and "
                "`uv run python -m ml.evaluation.find_failures`."
            )
        ),
    }
