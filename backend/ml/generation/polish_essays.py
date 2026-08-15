"""Generate the AI_POLISHED class with real hosted models (Groq).

This is the class that matters most, because it is the realistic case:

    human writes essay -> AI edits it -> final essay

Every polished sample keeps a pointer to the human original (``parent_id``) and a
side-by-side record is written to ``data/processed/polish_pairs.jsonl`` with the
measured word delta and overlap. That file is what makes the AI_POLISHED class
auditable — you can read exactly what the edit changed.

Group discipline: a polished essay is assigned **the same** ``group_id`` as the
human original, so the two can never land on opposite sides of a train/test
split. Without that rule the detector would be evaluated on paraphrases of essays
it had already memorised.

Usage
-----
    uv run python -m ml.generation.polish_essays --transforms clarity,formalize
    uv run python -m ml.generation.polish_essays --limit 20 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger, log_event
from ml.dataset_schema import Sample, dataset_paths, load_samples, write_jsonl
from ml.generation.groq_client import GroqClient, GroqUnavailableError
from ml.generation.prompts import POLISH_TRANSFORMS

logger = get_logger("ml.polish")


def _clean(text: str) -> str:
    out = text.strip()
    out = re.sub(
        r"^(sure[,!.]?\s*|certainly[,!.]?\s*|here(?:'s| is)[^\n]*\n+)",
        "",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(r"^```[a-z]*\n|\n```$", "", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def _overlap(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def polish(
    *,
    transforms: tuple[str, ...] | None = None,
    limit: int | None = None,
    replace: bool = False,
    data_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    settings = get_settings()
    data_dir = data_dir or settings.data_path
    paths = dataset_paths(data_dir)

    client = GroqClient()
    if not client.available:
        raise GroqUnavailableError(
            "GROQ_API_KEY is not set. Either export it, or build the offline corpus "
            "with `uv run python -m ml.generation.bootstrap_corpus`."
        )

    transform_names = transforms or tuple(POLISH_TRANSFORMS)
    humans = [s for s in load_samples(paths["human"]) if s.source == "seed_authored"]
    if not humans:
        raise FileNotFoundError(
            f"No human originals found in {paths['human']}. "
            "Run `uv run python -m ml.generation.bootstrap_corpus` first."
        )
    if limit:
        humans = humans[:limit]

    existing = [] if replace else load_samples(paths["ai_polished"])
    existing_ids = {s.record_id for s in existing}
    produced: list[Sample] = []
    pairs: list[dict[str, object]] = []
    failures = 0
    unchanged = 0

    models = client.models
    for index, human in enumerate(humans):
        for t_index, transform in enumerate(transform_names):
            # Rotate models so no transform is bound to a single model.
            model = models[(index + t_index) % len(models)]
            model_slug = re.sub(r"[^a-z0-9]+", "-", model.lower())
            record_id = f"{human.record_id}-groq-{transform}-{model_slug}"
            if record_id in existing_ids:
                continue

            spec = POLISH_TRANSFORMS[transform]
            if dry_run:
                log_event(
                    logger,
                    "polish.dry_run",
                    parent=human.record_id,
                    transform=transform,
                    model=model,
                )
                continue

            try:
                completion = client.complete(
                    model=model,
                    system=spec["system"],
                    user=spec["user"].format(text=human.text),
                    temperature=0.4,
                    top_p=1.0,
                    max_tokens=int(human.n_words * 2.5) + 300,
                )
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log_event(
                    logger,
                    "polish.failed",
                    level="warning",
                    parent=human.record_id,
                    transform=transform,
                    type=type(exc).__name__,
                )
                continue

            text = _clean(completion.text)
            overlap = _overlap(human.text, text)
            if text.strip() == human.text.strip():
                # An edit that changed nothing cannot be labelled AI_POLISHED —
                # the same string would carry two labels.
                unchanged += 1
                log_event(
                    logger,
                    "polish.no_change",
                    level="warning",
                    parent=human.record_id,
                    transform=transform,
                )
                continue
            if len(text.split()) < human.n_words * 0.4:
                failures += 1
                log_event(
                    logger,
                    "polish.truncated",
                    level="warning",
                    parent=human.record_id,
                    transform=transform,
                    words=len(text.split()),
                )
                continue

            produced.append(
                Sample(
                    record_id=record_id,
                    label="ai_polished",
                    text=text,
                    group_id=human.group_id,  # same group as the human original
                    source="groq",
                    topic=human.topic,
                    length_band=human.length_band,
                    voice=human.voice,
                    l2_english=human.l2_english,
                    model=model,
                    strategy=transform,
                    temperature=0.4,
                    top_p=1.0,
                    parent_id=human.record_id,
                    license="model-output-see-provider-terms",
                    notes=(
                        f"Groq {model} '{transform}' edit of {human.record_id} "
                        f"(expected change: {spec['expected_change']}, "
                        f"word overlap {overlap:.2f})."
                    ),
                )
            )
            pairs.append(
                {
                    "pair_id": record_id,
                    "seed_id": human.record_id,
                    "transform": transform,
                    "model": model,
                    "expected_change": spec["expected_change"],
                    "original": human.text,
                    "polished": text,
                    "similarity": round(overlap, 4),
                    "word_delta": len(text.split()) - human.n_words,
                }
            )
            log_event(
                logger,
                "polish.ok",
                parent=human.record_id,
                transform=transform,
                model=model,
                overlap=round(overlap, 3),
                word_delta=len(text.split()) - human.n_words,
            )

    if not dry_run:
        write_jsonl(paths["ai_polished"], [*existing, *produced])
        if pairs:
            mode = "w" if replace else "a"
            paths["pairs"].parent.mkdir(parents=True, exist_ok=True)
            with paths["pairs"].open(mode, encoding="utf-8") as handle:
                for pair in pairs:
                    handle.write(json.dumps(pair, ensure_ascii=False) + "\n")

    summary = {
        "polished": len(produced),
        "kept_existing": len(existing),
        "failures": failures,
        "unchanged_dropped": unchanged,
        "transforms": len(transform_names),
    }
    log_event(logger, "polish.complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create AI-polished essays via Groq.")
    parser.add_argument(
        "--transforms",
        type=str,
        default="",
        help=f"comma-separated subset of: {','.join(POLISH_TRANSFORMS)}",
    )
    parser.add_argument("--limit", type=int, default=None, help="limit human originals")
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    transforms = tuple(t.strip() for t in args.transforms.split(",") if t.strip()) or None
    if transforms:
        unknown = [t for t in transforms if t not in POLISH_TRANSFORMS]
        if unknown:
            raise SystemExit(f"Unknown transforms: {unknown}")

    summary = polish(
        transforms=transforms,
        limit=args.limit,
        replace=args.replace,
        dry_run=args.dry_run,
    )
    logger.info("polishing done | " + " ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
