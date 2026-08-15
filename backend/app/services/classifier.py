"""The trained classifiers - this is where the actual decision is made.

Nothing in this module consults a language model about authorship. It loads two
scikit-learn artifacts produced by ``ml/training/train.py`` and applies them:

``document_model``  three-class (human / ai_generated / ai_polished) over the
                    ~411-dimensional document vector. This produces the headline
                    verdict.
``sentence_model``  binary (human-like vs machine-like) over the ~189-dimensional
                    sentence vector, used for highlighting. It is trained only on
                    sentences from unambiguous documents - see
                    ``ml/training/extract_features.py``.

Both artifacts carry their own feature name list and column indices, so a model
trained on a subset of features (the ablation baselines) loads and runs through
exactly the same code path as the hybrid.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.exceptions import ModelNotTrainedError
from app.core.logging import get_logger, log_event

logger = get_logger("app.classifier")

DOCUMENT_ARTIFACT = "document_model.joblib"
SENTENCE_ARTIFACT = "sentence_model.joblib"
METADATA_FILE = "model_metadata.json"
REFERENCE_STATS_FILE = "reference_stats.json"
COMPARISON_FILE = "model_comparison.json"

CLASSES: tuple[str, ...] = ("human", "ai_generated", "ai_polished")


@dataclass(slots=True)
class LoadedModel:
    """One deserialised artifact plus the metadata needed to apply it."""

    kind: str
    name: str
    estimator: Any
    calibrated: Any | None
    feature_names: list[str]
    feature_indices: list[int]
    selected_names: list[str]
    classes: list[str]
    groups: tuple[str, ...]
    calibration_method: str | None = None
    coefficients: dict[str, Any] = field(default_factory=dict)

    @property
    def predictor(self) -> Any:
        """The calibrated model when one exists, otherwise the raw estimator."""
        return self.calibrated if self.calibrated is not None else self.estimator

    def vectorise(self, features: dict[str, float]) -> np.ndarray:
        """Build the model's input row from a feature dict.

        Missing keys become 0.0 rather than raising: a feature block can be
        legitimately empty (for example ``cor_*`` when the corpus reference is
        absent), and the scaler was fit with those same zeros.
        """
        full = np.array(
            [float(features.get(name, 0.0)) for name in self.feature_names],
            dtype=np.float64,
        )
        full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)
        return full[self.feature_indices].reshape(1, -1)

    def vectorise_many(self, rows: list[dict[str, float]]) -> np.ndarray:
        if not rows:
            return np.zeros((0, len(self.feature_indices)))
        matrix = np.array(
            [[float(r.get(name, 0.0)) for name in self.feature_names] for r in rows],
            dtype=np.float64,
        )
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix[:, self.feature_indices]


class DetectorModels:
    """Process-wide singleton holding the trained artifacts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.document: LoadedModel | None = None
        self.sentence: LoadedModel | None = None
        self.metadata: dict[str, Any] = {}
        self.reference_stats: dict[str, Any] = {}
        self.comparison: dict[str, Any] = {}
        self._loaded = False
        self._load_error: str | None = None

    # ------------------------------------------------------------- loading
    def load(self, artifacts_dir: Path | None = None, *, force: bool = False) -> bool:
        if self._loaded and not force:
            return True
        with self._lock:
            if self._loaded and not force:
                return True
            directory = Path(artifacts_dir or get_settings().artifacts_path)
            try:
                import joblib
            except ImportError as exc:  # pragma: no cover
                self._load_error = str(exc)
                return False

            doc_path = directory / DOCUMENT_ARTIFACT
            if not doc_path.exists():
                self._load_error = f"{DOCUMENT_ARTIFACT} not found in {directory}"
                log_event(
                    logger,
                    "models.not_trained",
                    level="warning",
                    directory=str(directory.name),
                    hint="run `uv run python -m ml.training.train`",
                )
                return False

            try:
                self.document = _as_loaded(joblib.load(doc_path))
                sent_path = directory / SENTENCE_ARTIFACT
                self.sentence = (
                    _as_loaded(joblib.load(sent_path)) if sent_path.exists() else None
                )
                self.metadata = _read_json(directory / METADATA_FILE)
                self.reference_stats = _read_json(directory / REFERENCE_STATS_FILE)
                self.comparison = _read_json(directory / COMPARISON_FILE)
                self._loaded = True
                self._load_error = None
                log_event(
                    logger,
                    "models.loaded",
                    document_model=self.document.name,
                    document_features=len(self.document.feature_indices),
                    sentence_model=self.sentence.name if self.sentence else "none",
                    model_version=self.metadata.get("model_version", "unknown"),
                    calibration=self.document.calibration_method or "none",
                )
                return True
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                log_event(
                    logger, "models.load_failed", level="error", type=type(exc).__name__
                )
                return False

    @property
    def ready(self) -> bool:
        return self._loaded and self.document is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def require(self) -> LoadedModel:
        if not self.ready:
            self.load()
        if not self.ready or self.document is None:
            raise ModelNotTrainedError(
                self._load_error
                or "The detector model has not been trained yet. "
                "Run `uv run python -m ml.training.train`."
            )
        return self.document

    # ---------------------------------------------------------- prediction
    def predict_document(self, features: dict[str, float]) -> dict[str, float]:
        """Calibrated class probabilities for one document."""
        model = self.require()
        row = model.vectorise(features)
        probabilities = model.predictor.predict_proba(row)[0]
        return {
            label: float(probabilities[i])
            for i, label in enumerate(model.classes)
        }

    def predict_sentences(self, rows: list[dict[str, float]]) -> list[float]:
        """Per-sentence machine-likeness in [0, 1].

        Returns an empty list when no sentence model is available; the caller
        surfaces that as "sentence-level scoring unavailable" rather than
        inventing scores.
        """
        if self.sentence is None or not rows:
            return []
        matrix = self.sentence.vectorise_many(rows)
        probabilities = self.sentence.predictor.predict_proba(matrix)
        ai_index = (
            self.sentence.classes.index("ai_generated")
            if "ai_generated" in self.sentence.classes
            else 1
        )
        return [float(p[ai_index]) for p in probabilities]

    def document_contributions(
        self, features: dict[str, float], *, top_k: int = 12
    ) -> list[dict[str, Any]]:
        """Per-feature contribution to the document decision.

        For a linear model this is exact: ``coef * standardised_value``. For a
        tree model we fall back to the global importances weighted by how far the
        value sits from the training mean, which is an approximation and is
        labelled as such in the returned ``method`` field.
        """
        model = self.require()
        return _contributions(model, features, top_k=top_k)

    def sentence_contributions(
        self, features: dict[str, float], *, top_k: int = 8
    ) -> list[dict[str, Any]]:
        if self.sentence is None:
            return []
        return _contributions(self.sentence, features, top_k=top_k)

    # ------------------------------------------------------------ metadata
    def info(self) -> dict[str, Any]:
        settings = get_settings()
        model = self.document
        return {
            "ready": self.ready,
            "error": self._load_error,
            "detector_version": settings.detector_version,
            "model_version": self.metadata.get("model_version"),
            "dataset_version": self.metadata.get("dataset_version"),
            "features_version": self.metadata.get("features_version"),
            "trained_at": self.metadata.get("trained_at"),
            "data_regime": self.metadata.get("data_regime"),
            "document_model": {
                "name": model.name if model else None,
                "n_features": len(model.feature_indices) if model else 0,
                "feature_groups": list(model.groups) if model else [],
                "calibration": model.calibration_method if model else None,
                "classes": list(model.classes) if model else [],
            },
            "sentence_model": {
                "name": self.sentence.name if self.sentence else None,
                "n_features": len(self.sentence.feature_indices) if self.sentence else 0,
                "calibration": self.sentence.calibration_method if self.sentence else None,
                "classes": list(self.sentence.classes) if self.sentence else [],
            },
            "language_model": {
                "name": settings.lm_model_name,
                "role": "feature instrument only",
            },
            "metrics": self.metadata.get("metrics", {}),
            "training": self.metadata.get("training", {}),
            "feature_importance": self.metadata.get("feature_importance", [])[:20],
            "model_comparison": self.comparison.get("models", []),
        }


def _as_loaded(payload: Any) -> LoadedModel:
    if isinstance(payload, LoadedModel):
        return payload
    return LoadedModel(
        kind=payload["kind"],
        name=payload["name"],
        estimator=payload["estimator"],
        calibrated=payload.get("calibrated"),
        feature_names=list(payload["feature_names"]),
        feature_indices=list(payload["feature_indices"]),
        selected_names=list(payload.get("selected_names", [])),
        classes=list(payload["classes"]),
        groups=tuple(payload.get("groups", ())),
        calibration_method=payload.get("calibration_method"),
        coefficients=payload.get("coefficients", {}),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:  # pragma: no cover
        log_event(logger, "models.bad_json", level="warning", file=path.name)
        return {}


def _contributions(
    model: LoadedModel, features: dict[str, float], *, top_k: int
) -> list[dict[str, Any]]:
    """Signed per-feature contributions, most influential first."""
    coefficients = model.coefficients or {}
    names: list[str] = coefficients.get("names") or model.selected_names
    if not names:
        return []

    means = np.asarray(coefficients.get("scaler_mean") or [], dtype=np.float64)
    scales = np.asarray(coefficients.get("scaler_scale") or [], dtype=np.float64)
    values = np.array([float(features.get(n, 0.0)) for n in names], dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if means.size == values.size and scales.size == values.size:
        with np.errstate(divide="ignore", invalid="ignore"):
            standardised = np.where(scales > 1e-12, (values - means) / scales, 0.0)
    else:
        standardised = values

    weights = coefficients.get("weights")
    method = coefficients.get("method", "unknown")
    if weights is None:
        return []

    weight_matrix = np.asarray(weights, dtype=np.float64)
    if weight_matrix.ndim == 1:
        weight_matrix = weight_matrix.reshape(1, -1)
    if weight_matrix.shape[1] != values.size:
        return []

    # Direction of interest: how much this feature pushes away from "human".
    classes = list(model.classes)
    if method == "linear" and len(classes) == weight_matrix.shape[0]:
        human_index = classes.index("human") if "human" in classes else 0
        machine_rows = [i for i in range(len(classes)) if i != human_index]
        signal = weight_matrix[machine_rows].mean(axis=0) - weight_matrix[human_index]
    elif method == "linear":
        signal = weight_matrix[0]
    else:
        signal = weight_matrix.mean(axis=0)

    contributions = signal * standardised
    order = np.argsort(-np.abs(contributions))[:top_k]
    return [
        {
            "feature": names[i],
            "value": round(float(values[i]), 6),
            "standardised": round(float(standardised[i]), 4),
            "contribution": round(float(contributions[i]), 5),
            "direction": "machine-like" if contributions[i] > 0 else "human-like",
            "method": method,
        }
        for i in order
        if abs(contributions[i]) > 1e-9
    ]


detector_models = DetectorModels()
