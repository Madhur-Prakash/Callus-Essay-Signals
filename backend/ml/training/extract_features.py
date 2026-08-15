"""Extract feature matrices for the whole corpus.

Two matrices are produced and cached in ``data/processed/features.npz``:

``X_doc``   (n_documents, ~411) the document vector for the three-class model
``X_sent``  (n_sentences,  ~189) the sentence vector for the highlighting model

Ordering discipline
-------------------
The corpus reference (``CorpusReference``) has to be fit on **training documents
only**, but the ``cor_*`` features are part of both matrices. So this runs in two
passes:

1. Pass 1 extracts everything except ``cor_*`` and collects the training
   documents' POS/function-word views.
2. The reference is fit on those training views and saved to the artifacts.
3. Pass 2 re-scores only the ``cor_*`` block for every document and sentence
   using the fitted reference.

The alternative — fitting the reference on all data — would leak test-set style
statistics into a training feature, which is exactly the kind of mistake that
produces a detector that looks excellent and generalises badly.

Sentence labels
---------------
Sentence rows inherit the document label, but ``sentence_trainable`` marks only
sentences from ``human`` and ``ai_generated`` documents. AI_POLISHED documents
have genuinely mixed sentence-level authorship — some sentences are untouched
human text — so using them as sentence-level supervision would inject label
noise. They are still scored at inference time; they are just not trained on.

Usage
-----
    uv run python -m ml.training.extract_features
    uv run python -m ml.training.extract_features --limit 40      # quick smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.config import get_settings
from app.core.logging import get_logger, log_event, set_logger_level
from app.services.corpus_analyzer import (
    COR_FEATURE_NAMES,
    CorpusReference,
    aggregate_document_views,
)
from app.services.feature_extractor import (
    FEATURES_VERSION,
    FeatureExtractor,
    document_feature_names,
    sentence_feature_names,
)
from ml.dataset_schema import (
    DATASET_VERSION,
    LABEL_TO_INDEX,
    Sample,
    dataset_paths,
    load_samples,
)

logger = get_logger("ml.features")

SENTENCE_TRAINABLE_LABELS = ("human", "ai_generated")


def extract_all(
    *,
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    artifacts_dir = artifacts_dir or settings.artifacts_path
    paths = dataset_paths(data_dir)

    samples = load_samples(paths["combined"])
    if not samples:
        raise FileNotFoundError(
            f"{paths['combined']} is empty. Run `uv run python -m ml.training.prepare_dataset`."
        )
    if limit:
        samples = samples[:limit]

    doc_names = list(document_feature_names())
    sent_names = list(sentence_feature_names())
    cor_index_doc = [doc_names.index(n) for n in COR_FEATURE_NAMES]
    cor_index_sent = [sent_names.index(n) for n in COR_FEATURE_NAMES]

    extractor = FeatureExtractor(corpus_reference=None)
    extractor.warmup()

    # The per-document pipeline logs two INFO lines each; over 600 documents that
    # buries the progress lines that actually matter. Batch mode raises them to
    # WARNING so real problems still surface.
    for name in ("app.features", "app.lm"):
        set_logger_level(name, logging.WARNING)

    started = time.perf_counter()

    # ------------------------------------------------------------- pass one
    doc_rows: list[np.ndarray] = []
    doc_meta: list[dict[str, Any]] = []
    sent_rows: list[np.ndarray] = []
    sent_meta: list[dict[str, Any]] = []
    train_views: list[dict[str, Any]] = []
    # Views cached from pass 1 so pass 2 never has to re-run the language model.
    doc_views_cache: list[dict[str, Any]] = []
    sent_views_cache: list[dict[str, Any]] = []
    skipped = 0

    for i, sample in enumerate(samples, start=1):
        try:
            result = extractor.extract(sample.text, with_corpus=False)
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            log_event(
                logger,
                "extract.failed",
                level="warning",
                record_id=sample.record_id,
                type=type(exc).__name__,
            )
            continue
        if not result.document.sentences:
            skipped += 1
            continue

        doc_rows.append(np.array([result.document_features[n] for n in doc_names], dtype=np.float32))
        doc_meta.append(_document_meta(sample, result))

        views = aggregate_document_views(result.document)
        doc_views_cache.append({"text": sample.text, **views})
        if sample.split == "train":
            train_views.append({"label": sample.label, "text": sample.text, **views})

        for sentence, features in zip(
            result.document.sentences, result.sentence_features, strict=False
        ):
            sent_rows.append(np.array([features[n] for n in sent_names], dtype=np.float32))
            sent_meta.append(
                {
                    "record_id": sample.record_id,
                    "document_index": len(doc_rows) - 1,
                    "sentence_index": sentence.index,
                    "paragraph_index": sentence.paragraph_index,
                    "label": sample.label,
                    "label_index": LABEL_TO_INDEX[sample.label],
                    "split": sample.split,
                    "group_id": sample.group_id,
                    "n_words": sentence.n_words,
                    "trainable": sample.label in SENTENCE_TRAINABLE_LABELS,
                }
            )
            sent_views_cache.append(
                {
                    "text": sentence.text,
                    "pos_sequence": sentence.pos_sequence,
                    "function_word_profile": sentence.function_word_profile,
                    "pos_distribution": sentence.pos_distribution,
                }
            )

        if i % 50 == 0 or i == len(samples):
            log_event(
                logger,
                "extract.progress",
                done=i,
                total=len(samples),
                sentences=len(sent_rows),
                elapsed_s=round(time.perf_counter() - started, 1),
            )

    if not doc_rows:
        raise RuntimeError("Feature extraction produced no rows.")

    X_doc = np.vstack(doc_rows)
    X_sent = np.vstack(sent_rows)

    # ------------------------------------------- fit the corpus reference
    reference = CorpusReference()
    if train_views:
        reference.fit(train_views, dataset_version=DATASET_VERSION)
        reference.save(Path(artifacts_dir))
    else:
        log_event(logger, "extract.no_train_views", level="warning")

    # ------------------------------------------------------------ pass two
    # Only the cor_* block is recomputed, from the views cached in pass 1 — no
    # re-segmentation and, critically, no second language-model pass.
    if reference.fitted:
        corpus_started = time.perf_counter()
        for doc_i, views in enumerate(doc_views_cache):
            doc_scores = reference.score(
                views["text"],
                pos_sequence=views["pos_sequence"],
                function_word_profile=views["function_word_profile"],
                pos_distribution=views["pos_distribution"],
            )
            for col, name in zip(cor_index_doc, COR_FEATURE_NAMES, strict=False):
                X_doc[doc_i, col] = doc_scores[name]

        for row, views in enumerate(sent_views_cache):
            scores = reference.score(
                views["text"],
                pos_sequence=views["pos_sequence"],
                function_word_profile=views["function_word_profile"],
                pos_distribution=views["pos_distribution"],
            )
            for col, name in zip(cor_index_sent, COR_FEATURE_NAMES, strict=False):
                X_sent[row, col] = scores[name]

        log_event(
            logger,
            "extract.corpus_scored",
            documents=len(doc_views_cache),
            sentences=len(sent_views_cache),
            elapsed_s=round(time.perf_counter() - corpus_started, 1),
        )

    # Guard against non-finite values before they reach a scaler.
    X_doc = np.nan_to_num(X_doc, nan=0.0, posinf=0.0, neginf=0.0)
    X_sent = np.nan_to_num(X_sent, nan=0.0, posinf=0.0, neginf=0.0)

    y_doc = np.array([m["label_index"] for m in doc_meta], dtype=np.int64)
    y_sent = np.array([m["label_index"] for m in sent_meta], dtype=np.int64)

    paths["features"].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paths["features"],
        X_doc=X_doc,
        y_doc=y_doc,
        X_sent=X_sent,
        y_sent=y_sent,
        doc_feature_names=np.array(doc_names),
        sent_feature_names=np.array(sent_names),
        doc_splits=np.array([m["split"] or "train" for m in doc_meta]),
        sent_splits=np.array([m["split"] or "train" for m in sent_meta]),
        doc_groups=np.array([m["group_id"] for m in doc_meta]),
        sent_groups=np.array([m["group_id"] for m in sent_meta]),
        sent_trainable=np.array([m["trainable"] for m in sent_meta]),
    )

    duration = round(time.perf_counter() - started, 2)
    manifest = {
        "features_version": FEATURES_VERSION,
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "lm_model": settings.lm_model_name,
        "spacy_model": settings.spacy_model,
        "documents": int(X_doc.shape[0]),
        "document_features": int(X_doc.shape[1]),
        "sentences": int(X_sent.shape[0]),
        "sentence_features": int(X_sent.shape[1]),
        "skipped_documents": skipped,
        "sentences_trainable": int(sum(1 for m in sent_meta if m["trainable"])),
        "extraction_seconds": duration,
        "corpus_reference": reference.summary(),
        "document_metadata": doc_meta,
        "notes": [
            "cor_* features are fit on the training split only; see module docstring.",
            "Sentence rows from ai_polished documents are excluded from sentence-level "
            "training (mixed authorship => noisy labels) but still scored at inference.",
        ],
    }
    paths["feature_manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    log_event(
        logger,
        "extract.complete",
        documents=int(X_doc.shape[0]),
        document_features=int(X_doc.shape[1]),
        sentences=int(X_sent.shape[0]),
        sentence_features=int(X_sent.shape[1]),
        skipped=skipped,
        duration_s=duration,
    )
    return {k: v for k, v in manifest.items() if k != "document_metadata"}


def _document_meta(sample: Sample, result) -> dict[str, Any]:  # noqa: ANN001
    return {
        "record_id": sample.record_id,
        "label": sample.label,
        "label_index": LABEL_TO_INDEX[sample.label],
        "split": sample.split,
        "group_id": sample.group_id,
        "topic": sample.topic,
        "model": sample.model,
        "strategy": sample.strategy,
        "length_band": sample.length_band,
        "l2_english": sample.l2_english,
        "source": sample.source,
        "voice": sample.voice,
        "n_words": sample.n_words,
        "n_sentences": result.n_sentences,
        "n_paragraphs": len(result.document.paragraphs),
        "parent_id": sample.parent_id,
        "segmentation_backend": result.backend,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features for the corpus.")
    parser.add_argument("--limit", type=int, default=None, help="only process N documents")
    args = parser.parse_args()

    summary = extract_all(limit=args.limit)
    logger.info(
        f"features extracted | documents={summary['documents']} x "
        f"{summary['document_features']} sentences={summary['sentences']} x "
        f"{summary['sentence_features']} in {summary['extraction_seconds']}s"
    )


if __name__ == "__main__":
    main()
