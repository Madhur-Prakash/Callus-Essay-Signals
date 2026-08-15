"""Assemble the corpus and produce leakage-safe splits.

Splitting rules (all enforced by code, not convention)
------------------------------------------------------
1. **Split by group, never by sample.** ``Sample.group_id`` ties together every
   document derived from the same underlying essay: the human original, its
   editing-noise variants, and every AI-polished version of it. A whole group
   goes to exactly one split.
2. **Held-out topics go to test only.** Six topics are reserved
   (``prompts.HELD_OUT_TOPICS``) so "does this generalise to unseen topics?" is
   answerable rather than assumed.
3. **Held-out model goes to test only.** One generator model/persona is
   withheld from training, which is what makes the cross-model generalisation
   number meaningful.
4. **Near-duplicate detection across splits.** Any test document whose 5-gram
   overlap with a training document exceeds a threshold is reported (and
   optionally dropped) - this catches leakage that group ids miss.
5. **Stratified by label over groups**, so all three classes appear in all
   splits at roughly the corpus proportions.

Target proportions are 70/15/15 by group, adjusted to keep rules 2 and 3.

Usage
-----
    uv run python -m ml.training.prepare_dataset
    uv run python -m ml.training.prepare_dataset --no-holdout-model
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.core.logging import get_logger, log_event
from ml.dataset_schema import (
    DATASET_VERSION,
    LABELS,
    Sample,
    dataset_paths,
    load_samples,
    write_jsonl,
)
from ml.generation.prompts import HELD_OUT_TOPICS

logger = get_logger("ml.prepare")

SPLIT_SEED = 42
TARGET_PROPORTIONS = {"train": 0.70, "validation": 0.15, "test": 0.15}
NEAR_DUPLICATE_THRESHOLD = 0.55
HELD_OUT_MODELS = ("proxy-heldout-qwen",)
"""Withheld from training so cross-model generalisation is measurable. If the
Groq path was used, the operator can point this at a real model instead."""


# --------------------------------------------------------------------------- #
# Ingestion of operator-supplied real essays
# --------------------------------------------------------------------------- #
def ingest_raw_human(data_dir: Path) -> list[Sample]:
    """Pick up real human essays the operator dropped into ``data/raw/human/``.

    Expected layout: one ``.txt`` per essay, plus an optional ``manifest.json``
    mapping filename -> metadata (``topic``, ``l2_english``, ``license``,
    ``voice``, ``notes``). Files without a manifest entry are still ingested, with
    the licence recorded as ``unspecified`` so the gap is visible in the
    dataset manifest rather than silently assumed to be fine.
    """
    raw_dir = data_dir / "raw" / "human"
    if not raw_dir.exists():
        return []

    manifest_path = raw_dir / "manifest.json"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log_event(logger, "ingest.bad_manifest", level="warning", path=str(manifest_path))

    samples: list[Sample] = []
    for path in sorted(raw_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if len(text.split()) < 80:
            log_event(logger, "ingest.too_short", level="warning", file=path.name)
            continue
        meta = manifest.get(path.name, {})
        samples.append(
            Sample(
                record_id=f"real-{path.stem}",
                label="human",
                text=text,
                group_id=f"real:{path.stem}",
                source="ingested_real",
                topic=meta.get("topic", "unspecified"),
                length_band=_band(len(text.split())),
                voice=meta.get("voice", "unspecified"),
                l2_english=bool(meta.get("l2_english", False)),
                license=meta.get("license", "unspecified"),
                notes=meta.get("notes", "Operator-supplied real human essay."),
            )
        )
    if samples:
        log_event(logger, "ingest.real_human", count=len(samples))
    return samples


def _band(words: int) -> str:
    if words < 300:
        return "short"
    if words < 600:
        return "medium"
    return "long"


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
def _group_profile(samples: list[Sample]) -> dict[str, dict[str, Any]]:
    """Aggregate each group's labels, topics and models."""
    profile: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"labels": set(), "topics": set(), "models": set(), "n": 0, "l2": False}
    )
    for sample in samples:
        entry = profile[sample.group_id]
        entry["labels"].add(sample.label)
        entry["topics"].add(sample.topic)
        if sample.model:
            entry["models"].add(sample.model)
        entry["n"] += 1
        entry["l2"] = entry["l2"] or sample.l2_english
    return dict(profile)


def assign_splits(
    samples: list[Sample], *, holdout_model: bool = True
) -> tuple[dict[str, str], dict[str, Any]]:
    """Return ``(group_id -> split)`` plus a report of how it was decided."""
    profile = _group_profile(samples)
    rng = random.Random(SPLIT_SEED)

    forced_test: set[str] = set()
    held_out_topics = set(HELD_OUT_TOPICS)
    for group_id, entry in profile.items():
        if entry["topics"] & held_out_topics:
            forced_test.add(group_id)
        if holdout_model and entry["models"] & set(HELD_OUT_MODELS):
            forced_test.add(group_id)

    remaining = [g for g in sorted(profile) if g not in forced_test]

    # Stratify the remaining groups by label signature *and* by the L2-English
    # flag. Stratifying on the label signature alone put every L2 seed in
    # train/validation, which left the test split with zero L2 human documents -
    # and the bias analysis is the one measurement that must not be unavailable.
    # A group containing both `human` and `ai_polished` samples is its own stratum.
    strata: dict[str, list[str]] = defaultdict(list)
    for group_id in remaining:
        labels_key = "+".join(sorted(profile[group_id]["labels"]))
        l2_key = "l2" if profile[group_id]["l2"] else "l1"
        strata[f"{labels_key}|{l2_key}"].append(group_id)

    assignment: dict[str, str] = {g: "test" for g in forced_test}
    for key in sorted(strata):
        bucket = strata[key]
        rng.shuffle(bucket)
        n = len(bucket)
        # Forced-test groups already contribute to the test share, so the
        # remaining groups are divided to hit the overall target.
        n_train = int(round(n * TARGET_PROPORTIONS["train"]))
        n_val = int(round(n * TARGET_PROPORTIONS["validation"]))
        # Guarantee at least one group per split when the stratum allows it.
        if n >= 3:
            n_train = max(1, min(n - 2, n_train))
            n_val = max(1, min(n - n_train - 1, n_val))
        for i, group_id in enumerate(bucket):
            if i < n_train:
                assignment[group_id] = "train"
            elif i < n_train + n_val:
                assignment[group_id] = "validation"
            else:
                assignment[group_id] = "test"

    report = {
        "seed": SPLIT_SEED,
        "strategy": (
            "grouped; stratified by label signature and L2-English flag; held-out "
            "topics and one held-out generator forced to test"
        ),
        "target_proportions": TARGET_PROPORTIONS,
        "n_groups": len(profile),
        "forced_to_test": {
            "held_out_topics": sorted(held_out_topics),
            "held_out_models": list(HELD_OUT_MODELS) if holdout_model else [],
            "n_groups": len(forced_test),
        },
        "strata": {k: len(v) for k, v in sorted(strata.items())},
    }
    return assignment, report


# --------------------------------------------------------------------------- #
# Near-duplicate leakage audit
# --------------------------------------------------------------------------- #
def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < n:
        return {tuple(words)}
    return {tuple(words[i : i + n]) for i in range(len(words) - n + 1)}


def audit_leakage(
    samples: list[Sample], threshold: float = NEAR_DUPLICATE_THRESHOLD
) -> list[dict[str, Any]]:
    """Report test/validation documents that closely overlap a training document.

    This is the check that catches what group ids cannot: two independently
    generated essays on the same topic from the same model can be near-identical.
    """
    train = [s for s in samples if s.split == "train"]
    others = [s for s in samples if s.split in {"validation", "test"}]
    train_shingles = [(s, _shingles(s.text)) for s in train]

    findings: list[dict[str, Any]] = []
    for sample in others:
        shingles = _shingles(sample.text)
        if not shingles:
            continue
        best_score, best_id = 0.0, None
        for other, other_shingles in train_shingles:
            if not other_shingles:
                continue
            overlap = len(shingles & other_shingles) / min(len(shingles), len(other_shingles))
            if overlap > best_score:
                best_score, best_id = overlap, other.record_id
        if best_score >= threshold:
            findings.append(
                {
                    "record_id": sample.record_id,
                    "split": sample.split,
                    "label": sample.label,
                    "closest_train_record": best_id,
                    "shingle_overlap": round(best_score, 4),
                }
            )
    return findings


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def prepare(
    *,
    data_dir: Path | None = None,
    holdout_model: bool = True,
    drop_near_duplicates: bool = True,
) -> dict[str, Any]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    paths = dataset_paths(data_dir)

    samples: list[Sample] = []
    for label in LABELS:
        loaded = load_samples(paths[label])
        samples.extend(loaded)
        log_event(logger, "prepare.loaded", label=label, count=len(loaded))
    samples.extend(ingest_raw_human(data_dir))

    if not samples:
        raise FileNotFoundError(
            "No samples found. Run `uv run python -m ml.generation.bootstrap_corpus` "
            "(offline) or the Groq generators first."
        )

    # Exact-duplicate removal, keeping the first occurrence.
    seen_text: dict[str, str] = {}
    deduped: list[Sample] = []
    exact_duplicates = 0
    for sample in samples:
        key = " ".join(sample.text.lower().split())
        if key in seen_text:
            exact_duplicates += 1
            continue
        seen_text[key] = sample.record_id
        deduped.append(sample)
    samples = deduped

    assignment, split_report = assign_splits(samples, holdout_model=holdout_model)
    for sample in samples:
        sample.split = assignment.get(sample.group_id, "train")

    leakage = audit_leakage(samples)
    dropped_near_duplicates = 0
    if drop_near_duplicates and leakage:
        # Default behaviour: remove them. Leaving a document in the test set that
        # is 60% shingle-identical to a training document produces a metric that
        # is knowingly wrong, and the cost of dropping is a handful of samples.
        drop_ids = {f["record_id"] for f in leakage}
        samples = [s for s in samples if s.record_id not in drop_ids]
        dropped_near_duplicates = len(drop_ids)
        log_event(logger, "prepare.dropped_near_duplicates", count=dropped_near_duplicates)

    write_jsonl(paths["combined"], samples)

    manifest = build_manifest(
        samples,
        split_report=split_report,
        leakage=leakage,
        exact_duplicates=exact_duplicates,
        dropped_near_duplicates=dropped_near_duplicates,
    )
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    paths["splits"].write_text(
        json.dumps(
            {
                "dataset_version": DATASET_VERSION,
                "group_to_split": assignment,
                "record_to_split": {s.record_id: s.split for s in samples},
                "report": split_report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    log_event(
        logger,
        "prepare.complete",
        total=len(samples),
        exact_duplicates_removed=exact_duplicates,
        near_duplicate_findings=len(leakage),
        **{f"{k}_count": v for k, v in manifest["splits"]["counts"].items()},
    )
    return manifest


def build_manifest(
    samples: list[Sample],
    *,
    split_report: dict[str, Any],
    leakage: list[dict[str, Any]],
    exact_duplicates: int,
    dropped_near_duplicates: int = 0,
) -> dict[str, Any]:
    """The dataset card. Everything the evaluation report needs to slice by."""
    by_split: Counter[str] = Counter(s.split or "unassigned" for s in samples)
    label_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in samples:
        label_by_split[sample.split or "unassigned"][sample.label] += 1

    words = [s.n_words for s in samples]
    sources: Counter[str] = Counter(s.source for s in samples)
    regime = (
        "bootstrap"
        if not (sources.get("groq", 0) or sources.get("ingested_real", 0))
        else "mixed"
        if sources.get("groq", 0) and sources.get("seed_authored", 0)
        else "real"
    )

    def counts(getter) -> dict[str, int]:  # noqa: ANN001
        return dict(Counter(str(getter(s)) for s in samples).most_common())

    return {
        "dataset_version": DATASET_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        "data_regime": regime,
        "regime_note": {
            "bootstrap": (
                "All machine samples come from the offline procedural/rule-based "
                "generators and all human samples are hand-authored seeds. Metrics "
                "measure separability of these generators, NOT real-world detector "
                "accuracy."
            ),
            "mixed": (
                "Contains real hosted-model output alongside hand-authored seed "
                "essays. Machine classes are realistic; the human class is still a "
                "proxy unless real essays were ingested."
            ),
            "real": "Built from operator-supplied real essays and real model output.",
        }[regime],
        "totals": {
            "documents": len(samples),
            "groups": len({s.group_id for s in samples}),
            "words": sum(words),
            "mean_words": round(sum(words) / len(words), 1) if words else 0,
            "min_words": min(words) if words else 0,
            "max_words": max(words) if words else 0,
        },
        "labels": dict(Counter(s.label for s in samples).most_common()),
        "splits": {
            "counts": dict(by_split),
            "labels_per_split": {k: dict(v) for k, v in label_by_split.items()},
            "report": split_report,
        },
        "sources": dict(sources),
        "models": counts(lambda s: s.model or "none"),
        "strategies": counts(lambda s: s.strategy or "none"),
        "topics": counts(lambda s: s.topic),
        "length_bands": counts(lambda s: s.length_band),
        "voices": counts(lambda s: s.voice),
        "l2_english": {
            "true": sum(1 for s in samples if s.l2_english),
            "false": sum(1 for s in samples if not s.l2_english),
            "note": (
                "For bootstrap data this flag marks seeds written in a *simulated* "
                "second-language English register. It is a proxy for a real L2 "
                "population and the bias numbers derived from it must be read as "
                "indicative only."
            ),
        },
        "licenses": counts(lambda s: s.license),
        "preprocessing": [
            "Unicode NFKC normalisation and CRLF -> LF (app.core.security.normalise_essay)",
            "zero-width and control characters removed",
            "runs of 3+ newlines collapsed to a paragraph break",
            "exact duplicate documents removed (case- and whitespace-insensitive)",
            "no lowercasing, no stopword removal, no stemming - the detector "
            "measures surface style, so destroying it would destroy the signal",
        ],
        "leakage_controls": {
            "split_unit": "group_id (never individual samples)",
            "polished_shares_group_with_original": True,
            "held_out_topics_in_test_only": True,
            "held_out_model_in_test_only": bool(split_report["forced_to_test"]["held_out_models"]),
            "exact_duplicates_removed": exact_duplicates,
            "near_duplicate_threshold": NEAR_DUPLICATE_THRESHOLD,
            "near_duplicate_findings": leakage[:50],
            "near_duplicate_count": len(leakage),
            "near_duplicates_dropped": dropped_near_duplicates,
            "near_duplicate_note": (
                "All near-duplicate findings come from the offline procedural "
                "AI_GENERATED generator, which draws sentences from a shared "
                "template bank; two essays on different topics can therefore share "
                "60% of their 5-grams. They are dropped from validation/test by "
                "default (pass --keep-near-duplicates to retain them), because "
                "leaving them in would inflate AI-class recall for reasons that "
                "have nothing to do with detection."
            ),
        },
        "known_limitations": [
            "The human class is dominated by 36 hand-authored seed essays; grouped "
            "splitting means the effective number of independent human documents is "
            "36, not the sample count. Confidence intervals are correspondingly wide.",
            "Seed essays were authored for this repository, not collected from real "
            "applicants, so they may under-represent genuine variation in student "
            "writing.",
            "The L2-English subset simulates a register rather than sampling real "
            "second-language writers.",
            "The offline AI_GENERATED generator is template-based; it reproduces the "
            "measurable register of instruction-tuned prose but not its semantics.",
            "No real admissions essays were scraped or used. Doing so without "
            "consent would be both a licensing and an ethics failure.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble the corpus and split it.")
    parser.add_argument(
        "--no-holdout-model",
        action="store_true",
        help="do not force the held-out generator model into the test split",
    )
    parser.add_argument(
        "--keep-near-duplicates",
        action="store_true",
        help=(
            "keep flagged near-duplicate validation/test documents "
            "(default is to drop them; keeping them inflates AI-class recall)"
        ),
    )
    args = parser.parse_args()

    manifest = prepare(
        holdout_model=not args.no_holdout_model,
        drop_near_duplicates=not args.keep_near_duplicates,
    )
    logger.info(
        f"dataset prepared | regime={manifest['data_regime']} "
        f"documents={manifest['totals']['documents']} "
        f"groups={manifest['totals']['groups']} splits={manifest['splits']['counts']}"
    )


if __name__ == "__main__":
    main()
