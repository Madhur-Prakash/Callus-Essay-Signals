"""Train the detector, run the ablation study, calibrate, and write artifacts.

Protocol
--------
1. **Model selection by grouped cross-validation on the training split only.**
   ``GroupKFold`` over ``group_id`` so an essay's variants never sit on both sides
   of a fold boundary. Selection metric is macro F1.
2. **Ablation study.** Four feature sets are trained under the identical protocol
   so the question "does the language model actually add anything?" gets a number
   instead of an assertion:

   ``baseline_stylometric``  surface + syntax + burstiness + repetition + structure
   ``lm_only``               language-model probability features alone
   ``hybrid_no_shift``       everything except within-document style shift
   ``hybrid``                everything

3. **Calibration on the validation split** (Platt scaling; see
   ``app.services.calibration.fit_calibrator`` for why not isotonic at this size).
4. **The test split is never touched here.** It is used only by
   ``ml/evaluation/evaluate.py``. Selecting on test would make every reported
   number meaningless.

Also trains the binary sentence-level model used for highlighting, and writes
``reference_stats.json`` — the per-class feature percentiles that the explanation
engine turns into English.

Usage
-----
    uv run python -m ml.training.train
    uv run python -m ml.training.train --calibration isotonic --seed 7
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger, log_event
from app.services.calibration import fit_calibrator
from app.services.classifier import (
    COMPARISON_FILE,
    DOCUMENT_ARTIFACT,
    METADATA_FILE,
    REFERENCE_STATS_FILE,
    SENTENCE_ARTIFACT,
    LoadedModel,
)
from app.services.feature_extractor import (
    FEATURES_VERSION,
    MODEL_FEATURE_SETS,
    select_columns,
)
from ml.dataset_schema import DATASET_VERSION, INDEX_TO_LABEL, LABELS, dataset_paths

logger = get_logger("ml.train")

MODEL_VERSION = "1.0.0"
RANDOM_SEED = 42
CV_SPLITS = 4

# Features that the explanation engine needs reference distributions for. Kept
# explicit so the artifact stays small and the UI's evidence lines are stable.
EXPLANATION_FEATURES: tuple[str, ...] = (
    "lm_mean_logprob",
    "lm_log_perplexity",
    "lm_perplexity",
    "lm_frac_top1",
    "lm_frac_top10",
    "lm_mean_entropy",
    "lm_mean_log_rank",
    "lm_frac_prob_gt_50",
    "lm_mean_top1_gap",
    "sty_n_words",
    "sty_root_ttr",
    "sty_ttr",
    "sty_mean_word_len",
    "sty_punct_density",
    "sty_comma_rate",
    "sty_contraction_rate",
    "sty_colloquial_rate",
    "sty_llm_phrase_rate",
    "sty_transition_word_rate",
    "sty_hedge_rate",
    "sty_first_person_rate",
    "sty_nominalization_rate",
    "syn_mean_dep_depth",
    "syn_n_clauses",
    "syn_pos_entropy",
    "syn_subordinate_ratio",
    "ctx_pos_js_to_doc",
    "ctx_style_distance_to_doc",
    "ctx_z_mean_logprob",
    "ctx_z_n_words",
    "cor_char_sim_delta_ai_human",
    "cor_pos_sim_delta_ai_human",
)

DOC_EXPLANATION_FEATURES: tuple[str, ...] = (
    "bur_cv_sent_len",
    "bur_std_sent_len",
    "bur_burstiness_index",
    "bur_mean_abs_adjacent_diff",
    "bur_normalised_len_entropy",
    "bur_frac_sent_within_20pct_of_mean",
    "bur_direction_changes",
    "rep_trigram_repeat_ratio",
    "rep_fourgram_repeat_ratio",
    "rep_pos_fourgram_repeat_ratio",
    "rep_max_pos_template_count",
    "rep_sentence_opener_repeat_ratio",
    "rep_transition_phrase_count",
    "rep_mean_sentence_jaccard",
    "shift_max_abs_z_logprob",
    "shift_max_style_distance_to_doc",
    "shift_n_changepoints",
    "shift_paragraph_logprob_std",
    "doc_words_per_sentence",
    "doc_flesch_reading_ease",
    "doc_cv_paragraph_words",
    "doc_type_token_ratio",
    "doc_root_type_token_ratio",
    "doc_hapax_ratio",
    "whole_lm_log_perplexity",
    "whole_lm_perplexity",
    "whole_lm_frac_top1",
    "whole_lm_mean_logprob",
    "whole_lm_mean_entropy",
    "rep_distinct3",
    "agg_mean_lm_mean_logprob",
    "agg_std_lm_mean_logprob",
    "agg_mean_lm_frac_top1",
    "agg_mean_lm_log_perplexity",
    "agg_std_sty_n_words",
    "agg_mean_sty_root_ttr",
    "agg_mean_sty_llm_phrase_rate",
    "agg_mean_sty_transition_word_rate",
    "agg_mean_sty_contraction_rate",
    "agg_mean_sty_colloquial_rate",
    "agg_mean_ctx_style_distance_to_doc",
    "agg_mean_syn_mean_dep_depth",
)


# --------------------------------------------------------------------------- #
# Estimator inventory
# --------------------------------------------------------------------------- #
def _build_estimators(seed: int, n_features: int) -> dict[str, Any]:
    """Candidate pipelines. All start with standardisation.

    With a few hundred training documents and hundreds of features, univariate
    pre-selection plus a strongly regularised linear model is the honest default;
    the tree models are included so the choice is made on evidence rather than
    taste.
    """
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    k = int(min(120, max(10, n_features)))

    return {
        "logreg_l2": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k)),
                (
                    "clf",
                    LogisticRegression(
                        C=0.3,
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "logreg_l2_strong": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("select", SelectKBest(f_classif, k=min(60, k))),
                (
                    "clf",
                    LogisticRegression(
                        C=0.05,
                        max_iter=4000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
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
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        num_leaves=15,
                        min_child_samples=10,
                        subsample=0.8,
                        subsample_freq=1,
                        colsample_bytree=0.6,
                        reg_lambda=1.0,
                        class_weight="balanced",
                        random_state=seed,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def _cross_validate(
    pipeline: Any, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> dict[str, float]:
    """Grouped CV. Returns macro F1, accuracy and log loss."""
    from sklearn.base import clone
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    from sklearn.model_selection import GroupKFold

    n_groups = len(np.unique(groups))
    n_splits = int(min(CV_SPLITS, max(2, n_groups)))
    splitter = GroupKFold(n_splits=n_splits)

    f1_scores: list[float] = []
    accuracies: list[float] = []
    losses: list[float] = []
    for train_idx, test_idx in splitter.split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = clone(pipeline)
        model.fit(X[train_idx], y[train_idx])
        predictions = model.predict(X[test_idx])
        f1_scores.append(f1_score(y[test_idx], predictions, average="macro", zero_division=0))
        accuracies.append(accuracy_score(y[test_idx], predictions))
        try:
            probabilities = model.predict_proba(X[test_idx])
            losses.append(
                log_loss(y[test_idx], probabilities, labels=list(range(len(LABELS))))
            )
        except Exception:  # pragma: no cover
            pass

    return {
        "cv_macro_f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
        "cv_macro_f1_std": float(np.std(f1_scores)) if f1_scores else 0.0,
        "cv_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "cv_log_loss": float(np.mean(losses)) if losses else float("nan"),
        "cv_folds": len(f1_scores),
    }


def _linear_coefficients(pipeline: Any, feature_names: list[str]) -> dict[str, Any]:
    """Extract the weights and scaler statistics needed for explanations."""
    scaler = pipeline.named_steps.get("scaler")
    selector = pipeline.named_steps.get("select")
    estimator = pipeline.named_steps.get("clf")

    names = list(feature_names)
    mean = getattr(scaler, "mean_", None)
    scale = getattr(scaler, "scale_", None)

    if selector is not None:
        mask = selector.get_support()
        names = [n for n, keep in zip(names, mask, strict=False) if keep]
        if mean is not None:
            mean = mean[mask]
        if scale is not None:
            scale = scale[mask]

    if hasattr(estimator, "coef_"):
        weights = np.asarray(estimator.coef_, dtype=np.float64)
        method = "linear"
    elif hasattr(estimator, "feature_importances_"):
        weights = np.asarray(estimator.feature_importances_, dtype=np.float64).reshape(1, -1)
        method = "tree_importance"
    else:  # pragma: no cover
        return {}

    return {
        "names": names,
        "weights": weights.tolist(),
        "method": method,
        "scaler_mean": mean.tolist() if mean is not None else [],
        "scaler_scale": scale.tolist() if scale is not None else [],
    }


def _feature_importance(
    pipeline: Any,
    feature_names: list[str],
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    *,
    top_k: int = 40,
) -> list[dict[str, Any]]:
    """Permutation importance on the validation split.

    Permutation importance is used rather than raw coefficients or Gini
    importance because it answers the question the dashboard actually asks — "how
    much does the model's accuracy depend on this measurement?" — and it is
    comparable across linear and tree models.
    """
    from sklearn.inspection import permutation_importance

    from app.services.feature_extractor import group_of

    if len(X_validation) < 10 or len(np.unique(y_validation)) < 2:
        return []
    try:
        result = permutation_importance(
            pipeline,
            X_validation,
            y_validation,
            n_repeats=8,
            random_state=seed,
            scoring="f1_macro",
            n_jobs=1,
        )
    except Exception:  # pragma: no cover
        return []

    order = np.argsort(-result.importances_mean)[:top_k]
    return [
        {
            "feature": feature_names[i],
            "group": group_of(feature_names[i]),
            "importance": round(float(result.importances_mean[i]), 6),
            "std": round(float(result.importances_std[i]), 6),
        }
        for i in order
        if result.importances_mean[i] > 0
    ]


# --------------------------------------------------------------------------- #
# Reference statistics for the explanation engine
# --------------------------------------------------------------------------- #
def _reference_stats(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    wanted: tuple[str, ...],
    scope: str,
) -> dict[str, Any]:
    """Per-class percentile tables for a curated feature list."""
    index = {name: i for i, name in enumerate(feature_names)}
    stats: dict[str, Any] = {}
    percentiles = (5, 10, 25, 50, 75, 90, 95)

    for name in wanted:
        if name not in index:
            continue
        column = X[:, index[name]]
        entry: dict[str, Any] = {"overall": _describe(column, percentiles)}
        for label_index, label in INDEX_TO_LABEL.items():
            mask = y == label_index
            if mask.sum() >= 5:
                entry[label] = _describe(column[mask], percentiles)
        stats[name] = entry

    return {"scope": scope, "n_samples": int(X.shape[0]), "features": stats}


def _describe(values: np.ndarray, percentiles: tuple[int, ...]) -> dict[str, float]:
    return {
        "mean": round(float(np.mean(values)), 6),
        "std": round(float(np.std(values)), 6),
        **{f"p{p}": round(float(np.percentile(values, p)), 6) for p in percentiles},
    }


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    calibration: str = "sigmoid",
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    artifacts_dir = Path(artifacts_dir or settings.artifacts_path)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(data_dir)

    if not paths["features"].exists():
        raise FileNotFoundError(
            f"{paths['features']} not found. Run "
            "`uv run python -m ml.training.extract_features` first."
        )

    bundle = np.load(paths["features"], allow_pickle=False)
    X_doc = bundle["X_doc"].astype(np.float64)
    y_doc = bundle["y_doc"]
    doc_names = [str(n) for n in bundle["doc_feature_names"]]
    doc_splits = np.array([str(s) for s in bundle["doc_splits"]])
    doc_groups = np.array([str(g) for g in bundle["doc_groups"]])

    X_sent = bundle["X_sent"].astype(np.float64)
    y_sent = bundle["y_sent"]
    sent_names = [str(n) for n in bundle["sent_feature_names"]]
    sent_splits = np.array([str(s) for s in bundle["sent_splits"]])
    sent_groups = np.array([str(g) for g in bundle["sent_groups"]])
    sent_trainable = bundle["sent_trainable"].astype(bool)

    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    started = time.perf_counter()

    train_mask = doc_splits == "train"
    val_mask = doc_splits == "validation"

    # Zero-variance columns make SelectKBest emit a warning per fold. They carry no
    # information by definition, so record which ones they are (useful signal about
    # the corpus) and stop the warning from burying the CV output.
    constant_doc_features = [
        doc_names[i]
        for i in range(X_doc.shape[1])
        if float(np.std(X_doc[train_mask][:, i])) < 1e-12
    ]
    if constant_doc_features:
        log_event(
            logger,
            "train.constant_features",
            level="warning",
            count=len(constant_doc_features),
            examples=",".join(constant_doc_features[:6]),
        )
    warnings.filterwarnings("ignore", message=".*are constant.*", category=UserWarning)
    warnings.filterwarnings("ignore", message=".*invalid value encountered in divide.*")
    log_event(
        logger,
        "train.data",
        documents=int(X_doc.shape[0]),
        features=int(X_doc.shape[1]),
        train=int(train_mask.sum()),
        validation=int(val_mask.sum()),
        test=int((doc_splits == "test").sum()),
        sentences=int(X_sent.shape[0]),
        trainable_sentences=int((sent_trainable & (sent_splits == "train")).sum()),
    )
    if train_mask.sum() < 30:
        raise RuntimeError(
            f"Only {train_mask.sum()} training documents. Generate more data before training."
        )

    # ------------------------------------------------- ablation / comparison
    comparison: list[dict[str, Any]] = []
    trained: dict[str, dict[str, Any]] = {}

    for set_name, groups in MODEL_FEATURE_SETS.items():
        indices = select_columns(doc_names, groups)
        if not indices:
            log_event(logger, "train.empty_feature_set", level="warning", set=set_name)
            continue
        selected_names = [doc_names[i] for i in indices]
        X_train = X_doc[train_mask][:, indices]
        y_train = y_doc[train_mask]
        g_train = doc_groups[train_mask]

        best: dict[str, Any] | None = None
        for estimator_name, pipeline in _build_estimators(seed, len(indices)).items():
            scores = _cross_validate(pipeline, X_train, y_train, g_train, seed)
            entry = {
                "feature_set": set_name,
                "estimator": estimator_name,
                "n_features": len(indices),
                "feature_groups": list(groups),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in scores.items()},
            }
            comparison.append(entry)
            log_event(
                logger,
                "train.cv",
                feature_set=set_name,
                estimator=estimator_name,
                macro_f1=round(scores["cv_macro_f1"], 4),
                accuracy=round(scores["cv_accuracy"], 4),
            )
            if best is None or scores["cv_macro_f1"] > best["scores"]["cv_macro_f1"]:
                best = {
                    "estimator_name": estimator_name,
                    "pipeline": pipeline,
                    "scores": scores,
                }

        assert best is not None
        from sklearn.base import clone

        fitted = clone(best["pipeline"])
        fitted.fit(X_train, y_train)
        trained[set_name] = {
            "estimator_name": best["estimator_name"],
            "pipeline": fitted,
            "indices": indices,
            "selected_names": selected_names,
            "groups": groups,
            "cv": best["scores"],
        }

    if "hybrid" not in trained:
        raise RuntimeError("The hybrid feature set failed to train; cannot continue.")

    # The hybrid model is the shipped detector. It is selected by design (it is
    # the system being built), and the ablation table reports whether that design
    # choice is justified — see ml/evaluation/evaluate.py for the test-set numbers.
    chosen = trained["hybrid"]
    # Calibration and permutation importance both use the validation split; the
    # chosen pipeline is already fitted on train.
    X_val_h = X_doc[val_mask][:, chosen["indices"]]
    y_val_h = y_doc[val_mask]

    calibrated, calibration_used = fit_calibrator(
        chosen["pipeline"], X_val_h, y_val_h, method=calibration
    )
    log_event(
        logger,
        "train.calibrated",
        method=calibration_used,
        calibration_samples=int(val_mask.sum()),
    )

    importance = _feature_importance(
        chosen["pipeline"], chosen["selected_names"], X_val_h, y_val_h, seed
    )

    document_model = LoadedModel(
        kind="document",
        name=f"hybrid::{chosen['estimator_name']}",
        estimator=chosen["pipeline"],
        calibrated=calibrated if calibration_used != "none" else None,
        feature_names=doc_names,
        feature_indices=chosen["indices"],
        selected_names=chosen["selected_names"],
        classes=list(LABELS),
        groups=tuple(chosen["groups"]),
        calibration_method=calibration_used,
        coefficients=_linear_coefficients(chosen["pipeline"], chosen["selected_names"]),
    )

    # ------------------------------------------------------- sentence model
    sentence_model, sentence_report = _train_sentence_model(
        X_sent,
        y_sent,
        sent_names,
        sent_splits,
        sent_groups,
        sent_trainable,
        seed=seed,
        calibration=calibration,
    )

    # ------------------------------------------------------------ artifacts
    import joblib

    joblib.dump(_serialise(document_model), artifacts_dir / DOCUMENT_ARTIFACT, compress=3)
    if sentence_model is not None:
        joblib.dump(_serialise(sentence_model), artifacts_dir / SENTENCE_ARTIFACT, compress=3)

    # Reference distributions come from the *training* split only, so the
    # explanation engine never describes a live essay relative to test data.
    reference = {
        "sentence": _reference_stats(
            X_sent[(sent_splits == "train")],
            y_sent[(sent_splits == "train")],
            sent_names,
            EXPLANATION_FEATURES,
            "sentence/train",
        ),
        "document": _reference_stats(
            X_doc[train_mask],
            y_doc[train_mask],
            doc_names,
            DOC_EXPLANATION_FEATURES + EXPLANATION_FEATURES,
            "document/train",
        ),
    }
    (artifacts_dir / REFERENCE_STATS_FILE).write_text(
        json.dumps(reference, indent=2), encoding="utf-8"
    )

    comparison_payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "protocol": (
            f"GroupKFold(n_splits<={CV_SPLITS}) over group_id on the training split; "
            "selection metric macro F1; the test split is untouched here."
        ),
        "models": sorted(comparison, key=lambda e: -e["cv_macro_f1"]),
        "selected": {
            "feature_set": "hybrid",
            "estimator": chosen["estimator_name"],
            "reason": (
                "The hybrid feature set is the system under construction; the "
                "estimator within it was chosen by grouped CV macro F1. Note that "
                "the shipped model is NOT simply the top row of the table — see "
                "'findings' for why."
            ),
        },
        "findings": _ablation_findings(trained),
        "constant_features": {
            "count": len(constant_doc_features),
            "names": constant_doc_features,
            "note": (
                "Zero variance across the training split, so they carry no "
                "information. Most are punctuation rates (semicolons, ellipses) that "
                "simply never occur in this corpus."
            ),
        },
        "best_per_feature_set": {
            name: {
                "estimator": info["estimator_name"],
                "cv_macro_f1": round(info["cv"]["cv_macro_f1"], 4),
                "cv_macro_f1_std": round(info["cv"]["cv_macro_f1_std"], 4),
                "cv_accuracy": round(info["cv"]["cv_accuracy"], 4),
                "n_features": len(info["indices"]),
            }
            for name, info in trained.items()
        },
    }
    (artifacts_dir / COMPARISON_FILE).write_text(
        json.dumps(comparison_payload, indent=2), encoding="utf-8"
    )

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_version": DATASET_VERSION,
        "features_version": FEATURES_VERSION,
        "detector_version": settings.detector_version,
        "trained_at": datetime.now(UTC).isoformat(),
        "data_regime": manifest.get("data_regime"),
        "data_regime_note": manifest.get("regime_note"),
        "training": {
            "seed": seed,
            "cv": f"GroupKFold(n_splits<={CV_SPLITS}) on group_id",
            "selection_metric": "macro F1",
            "calibration": calibration_used,
            "calibration_split": "validation",
            "n_train_documents": int(train_mask.sum()),
            "n_validation_documents": int(val_mask.sum()),
            "n_test_documents": int((doc_splits == "test").sum()),
            "n_train_sentences": int((sent_trainable & (sent_splits == "train")).sum()),
            "document_features_total": len(doc_names),
            "document_features_used": len(chosen["indices"]),
            "sentence_features_used": (
                len(sentence_model.feature_indices) if sentence_model else 0
            ),
            "lm_model": settings.lm_model_name,
            "spacy_model": settings.spacy_model,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "duration_seconds": round(time.perf_counter() - started, 2),
        },
        "document_model": {
            "name": document_model.name,
            "estimator": chosen["estimator_name"],
            "classes": list(LABELS),
        },
        "sentence_model": sentence_report,
        "metrics": {
            "note": (
                "Cross-validation numbers on the training split only. Held-out test "
                "metrics live in ml/evaluation/evaluation_report.json — run "
                "`uv run python -m ml.evaluation.evaluate`."
            ),
            "cv": {
                name: round(info["cv"]["cv_macro_f1"], 4) for name, info in trained.items()
            },
        },
        "feature_importance": importance,
        "artifacts": {
            "document_model": DOCUMENT_ARTIFACT,
            "sentence_model": SENTENCE_ARTIFACT if sentence_model else None,
            "corpus_reference": "corpus_reference.joblib",
            "reference_stats": REFERENCE_STATS_FILE,
            "model_comparison": COMPARISON_FILE,
        },
    }
    (artifacts_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    log_event(
        logger,
        "train.complete",
        model=document_model.name,
        calibration=calibration_used,
        hybrid_cv_macro_f1=round(trained["hybrid"]["cv"]["cv_macro_f1"], 4),
        baseline_cv_macro_f1=round(
            trained.get("baseline_stylometric", {}).get("cv", {}).get("cv_macro_f1", 0.0), 4
        ),
        lm_only_cv_macro_f1=round(
            trained.get("lm_only", {}).get("cv", {}).get("cv_macro_f1", 0.0), 4
        ),
        duration_s=round(time.perf_counter() - started, 2),
    )
    return metadata


def _ablation_findings(trained: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Answer the ablation questions with numbers, including inconvenient ones.

    Differences are reported alongside the CV standard deviation so a gap that
    sits inside fold-to-fold noise is not written up as a result.
    """
    def score(name: str) -> tuple[float, float]:
        info = trained.get(name)
        if not info:
            return 0.0, 0.0
        return info["cv"]["cv_macro_f1"], info["cv"]["cv_macro_f1_std"]

    hybrid, hybrid_std = score("hybrid")
    baseline, baseline_std = score("baseline_stylometric")
    lm_only, lm_std = score("lm_only")
    no_shift, no_shift_std = score("hybrid_no_shift")

    def verdict(delta: float, noise: float) -> str:
        if abs(delta) <= noise:
            return "no measurable difference (gap is inside fold-to-fold noise)"
        return "improves" if delta > 0 else "hurts"

    return [
        {
            "question": "Does combining LM probability and stylometric features beat stylometry alone?",
            "comparison": "hybrid vs baseline_stylometric",
            "macro_f1": {"hybrid": round(hybrid, 4), "baseline": round(baseline, 4)},
            "delta": round(hybrid - baseline, 4),
            "cv_std": {"hybrid": round(hybrid_std, 4), "baseline": round(baseline_std, 4)},
            "verdict": verdict(hybrid - baseline, max(hybrid_std, baseline_std)),
        },
        {
            "question": "Are language-model probability features sufficient on their own?",
            "comparison": "lm_only vs baseline_stylometric",
            "macro_f1": {"lm_only": round(lm_only, 4), "baseline": round(baseline, 4)},
            "delta": round(lm_only - baseline, 4),
            "cv_std": {"lm_only": round(lm_std, 4), "baseline": round(baseline_std, 4)},
            "verdict": verdict(lm_only - baseline, max(lm_std, baseline_std)),
        },
        {
            "question": "Does within-document style shift improve document classification?",
            "comparison": "hybrid vs hybrid_no_shift",
            "macro_f1": {"hybrid": round(hybrid, 4), "hybrid_no_shift": round(no_shift, 4)},
            "delta": round(hybrid - no_shift, 4),
            "cv_std": {"hybrid": round(hybrid_std, 4), "hybrid_no_shift": round(no_shift_std, 4)},
            "verdict": verdict(hybrid - no_shift, max(hybrid_std, no_shift_std)),
            "note": (
                "The style-shift block is retained in the shipped model even where it "
                "does not improve the document-level score, because the per-paragraph "
                "style-shift analysis is a user-facing output in its own right: it is "
                "what lets the UI point at the specific passage that differs from the "
                "rest of the essay. Dropping the features would remove that evidence "
                "to chase a difference this corpus cannot resolve."
            ),
        },
    ]


def _train_sentence_model(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    splits: np.ndarray,
    groups: np.ndarray,
    trainable: np.ndarray,
    *,
    seed: int,
    calibration: str,
) -> tuple[LoadedModel | None, dict[str, Any]]:
    """Binary human vs machine sentence scorer, used for highlighting.

    Trained only on sentences from ``human`` and ``ai_generated`` documents. Rows
    from ``ai_polished`` documents are excluded because their sentence-level
    authorship is genuinely mixed — some sentences in a polished essay are
    untouched human text, so treating them all as "machine" would teach the model
    that ordinary human sentences are machine-like, which is precisely the failure
    mode that produces false positives on real students.
    """
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    human_index = LABELS.index("human")
    ai_index = LABELS.index("ai_generated")

    train_mask = trainable & (splits == "train")
    val_mask = trainable & (splits == "validation")
    if train_mask.sum() < 100:
        log_event(
            logger,
            "train.sentence_skipped",
            level="warning",
            trainable=int(train_mask.sum()),
        )
        return None, {"trained": False, "reason": "not enough trainable sentences"}

    def binarise(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keep = mask & ((y == human_index) | (y == ai_index))
        return (
            X[keep],
            (y[keep] == ai_index).astype(np.int64),
            groups[keep],
        )

    X_train, y_train, g_train = binarise(train_mask)
    X_val, y_val, _ = binarise(val_mask)

    # Strong regularisation (C=0.05) is deliberate. On the bootstrap corpus the two
    # sentence classes are almost linearly separable, and a weakly regularised
    # logistic regression responds by driving coefficients toward infinity: it
    # converges slowly, produces saturated 0/1 probabilities, and generalises
    # worse to real text. A small C keeps the decision function smooth and the
    # probabilities usable.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=0.05,
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = _cross_validate(pipeline, X_train, y_train, g_train, seed)

    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)

    calibrated, method = (fitted, "none")
    validation_metrics: dict[str, float] = {}
    if len(X_val) >= 40 and len(np.unique(y_val)) == 2:
        calibrated, method = fit_calibrator(fitted, X_val, y_val, method=calibration)
        predictor = calibrated if method != "none" else fitted
        probabilities = predictor.predict_proba(X_val)[:, 1]
        validation_metrics = {
            "roc_auc": round(float(roc_auc_score(y_val, probabilities)), 4),
            "f1_at_0.5": round(
                float(f1_score(y_val, (probabilities >= 0.5).astype(int), zero_division=0)), 4
            ),
        }

    model = LoadedModel(
        kind="sentence",
        name="sentence_logreg",
        estimator=fitted,
        calibrated=calibrated if method != "none" else None,
        feature_names=names,
        feature_indices=list(range(len(names))),
        selected_names=list(names),
        classes=["human", "ai_generated"],
        groups=("lm", "stylometric", "syntactic", "style_shift", "corpus"),
        calibration_method=method,
        coefficients=_linear_coefficients(fitted, names),
    )
    report = {
        "trained": True,
        "name": model.name,
        "n_train_sentences": int(len(X_train)),
        "n_validation_sentences": int(len(X_val)),
        "classes": ["human", "ai_generated"],
        "excluded_from_training": "sentences from ai_polished documents (mixed authorship)",
        "calibration": method,
        "cv": {k: round(v, 4) if isinstance(v, float) else v for k, v in cv.items()},
        "validation": validation_metrics,
    }

    # A near-perfect score at the sentence level is a warning sign, not a result.
    # It means the two sentence populations in the training corpus are trivially
    # separable — which is true of the bootstrap generators and will NOT be true of
    # real essays. Recording it here keeps it from being read as a capability.
    auc = validation_metrics.get("roc_auc")
    if auc is not None and auc >= 0.99:
        report["separability_warning"] = (
            f"Validation ROC-AUC is {auc:.4f}. A score this high means the two "
            "sentence populations in this corpus are almost perfectly separable, "
            "which is a property of the training data rather than evidence that "
            "sentence-level detection is solved. On the offline bootstrap corpus the "
            "machine sentences come from a finite template bank, so surface and "
            "corpus-similarity features separate them almost completely. Expect a "
            "substantially lower and more useful score once real model output "
            "replaces the bootstrap classes."
        )
        log_event(
            logger,
            "train.sentence_separability_warning",
            level="warning",
            roc_auc=auc,
            hint="bootstrap corpus is trivially separable at the sentence level",
        )
    log_event(
        logger,
        "train.sentence_model",
        sentences=int(len(X_train)),
        cv_macro_f1=round(cv["cv_macro_f1"], 4),
        calibration=method,
        **{f"val_{k}": v for k, v in validation_metrics.items()},
    )
    return model, report


def _serialise(model: LoadedModel) -> dict[str, Any]:
    return {
        "kind": model.kind,
        "name": model.name,
        "estimator": model.estimator,
        "calibrated": model.calibrated,
        "feature_names": model.feature_names,
        "feature_indices": model.feature_indices,
        "selected_names": model.selected_names,
        "classes": model.classes,
        "groups": model.groups,
        "calibration_method": model.calibration_method,
        "coefficients": model.coefficients,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the detector.")
    parser.add_argument(
        "--calibration",
        choices=("sigmoid", "isotonic", "none"),
        default="sigmoid",
        help="calibration method fit on the validation split",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    metadata = train(calibration=args.calibration, seed=args.seed)
    logger.info(
        f"training complete | model={metadata['document_model']['name']} "
        f"calibration={metadata['training']['calibration']} "
        f"cv={metadata['metrics']['cv']}"
    )


if __name__ == "__main__":
    main()
