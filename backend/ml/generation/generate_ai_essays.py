"""Generate the AI_GENERATED class with real hosted models (Groq).

Deliberately varied along five axes so the dataset does not encode one model's
fingerprint: **model family**, **prompt strategy**, **topic**, **length band**
and **sampling temperature**. One of the strategies (``anti_detection``)
explicitly instructs the model to vary sentence length and avoid the usual
giveaway vocabulary - without adversarial samples the reported accuracy would be
optimistic in a way that matters.

Usage
-----
    # append to whatever is already in data/ai_generated/
    uv run python -m ml.generation.generate_ai_essays --per-model 30

    # replace the class entirely
    uv run python -m ml.generation.generate_ai_essays --per-model 30 --replace
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

from app.config import get_settings
from app.core.logging import get_logger, log_event
from ml.dataset_schema import Sample, dataset_paths, load_samples, write_jsonl
from ml.generation.groq_client import GroqClient, GroqUnavailableError
from ml.generation.prompts import (
    GENERATION_STRATEGIES,
    HELD_OUT_TOPICS,
    LENGTH_BANDS,
    TEMPERATURES,
    TOP_P_VALUES,
    TRAIN_TOPICS,
)

logger = get_logger("ml.generate")
GLOBAL_SEED = 913_411


def _clean(text: str) -> str:
    """Strip the wrapper text models add around a requested artefact."""
    out = text.strip()
    # Remove a leading "Here is ..." / "Sure! ..." line.
    out = re.sub(
        r"^(sure[,!.]?\s*|certainly[,!.]?\s*|here(?:'s| is)[^\n]*\n+)",
        "",
        out,
        flags=re.IGNORECASE,
    )
    # Remove a markdown title and code fences.
    out = re.sub(r"^#{1,6}\s+.*\n+", "", out)
    out = re.sub(r"^```[a-z]*\n|\n```$", "", out)
    # Remove a trailing "Word count: 512" style note.
    out = re.sub(r"\n+\s*[\(\[]?word count[:\s].*$", "", out, flags=re.IGNORECASE)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def generate(
    *,
    per_model: int = 24,
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

    rng = random.Random(GLOBAL_SEED)
    topics = list(TRAIN_TOPICS) + list(HELD_OUT_TOPICS)
    strategies = list(GENERATION_STRATEGIES)
    bands = list(LENGTH_BANDS)

    existing = [] if replace else load_samples(paths["ai_generated"])
    existing_ids = {s.record_id for s in existing}
    produced: list[Sample] = []
    failures = 0

    for model in client.models:
        for i in range(per_model):
            topic = topics[(i * 7 + hash(model) % 5) % len(topics)]
            strategy = strategies[i % len(strategies)]
            band = bands[i % len(bands)]
            low, high = LENGTH_BANDS[band]
            words = rng.randint(low, high)
            temperature = TEMPERATURES[i % len(TEMPERATURES)]
            top_p = TOP_P_VALUES[i % len(TOP_P_VALUES)]

            slug = re.sub(r"[^a-z0-9]+", "-", topic.lower())[:28].strip("-")
            model_slug = re.sub(r"[^a-z0-9]+", "-", model.lower())
            record_id = f"groq-{model_slug}-{strategy}-{slug}-{i}"
            if record_id in existing_ids:
                continue

            spec = GENERATION_STRATEGIES[strategy]
            if dry_run:
                log_event(
                    logger,
                    "generate.dry_run",
                    model=model,
                    strategy=strategy,
                    topic_slug=slug,
                    words=words,
                    temperature=temperature,
                )
                continue

            try:
                completion = client.complete(
                    model=model,
                    system=spec["system"],
                    user=spec["user"].format(topic=topic, words=words),
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=int(words * 2.2) + 200,
                )
            except Exception as exc:  # noqa: BLE001 - logged and skipped
                failures += 1
                log_event(
                    logger,
                    "generate.failed",
                    level="warning",
                    model=model,
                    strategy=strategy,
                    type=type(exc).__name__,
                )
                continue

            text = _clean(completion.text)
            if len(text.split()) < 120:
                failures += 1
                log_event(
                    logger,
                    "generate.too_short",
                    level="warning",
                    model=model,
                    words=len(text.split()),
                )
                continue

            produced.append(
                Sample(
                    record_id=record_id,
                    label="ai_generated",
                    text=text,
                    # Every generated essay is its own group: nothing else in the
                    # corpus derives from it.
                    group_id=f"groq:{model_slug}:{strategy}:{slug}:{i}",
                    source="groq",
                    topic=topic,
                    length_band=band,
                    voice=f"model:{strategy}",
                    model=model,
                    strategy=strategy,
                    temperature=temperature,
                    top_p=top_p,
                    license="model-output-see-provider-terms",
                    notes=(
                        f"Groq {model}, strategy={strategy}, requested~{words} words, "
                        f"T={temperature}, top_p={top_p}."
                    ),
                )
            )
            log_event(
                logger,
                "generate.ok",
                model=model,
                strategy=strategy,
                words=len(text.split()),
                latency_ms=completion.latency_ms,
            )

    if not dry_run:
        write_jsonl(paths["ai_generated"], [*existing, *produced])

    summary = {
        "generated": len(produced),
        "kept_existing": len(existing),
        "failures": failures,
        "models": len(client.models),
    }
    log_event(logger, "generate.complete", **summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI essays via Groq.")
    parser.add_argument("--per-model", type=int, default=24)
    parser.add_argument("--replace", action="store_true", help="discard existing samples")
    parser.add_argument("--dry-run", action="store_true", help="log the plan, call nothing")
    args = parser.parse_args()

    summary = generate(per_model=args.per_model, replace=args.replace, dry_run=args.dry_run)
    logger.info("ai generation done | " + " ".join(f"{k}={v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
