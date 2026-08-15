"""Dataset record schema and JSONL I/O.

Every sample in the corpus — regardless of which generator produced it — carries
the same provenance block, because the evaluation report slices by ``topic``,
``model``, ``length_band``, ``l2_english`` and ``source``. A sample without
provenance is a sample we cannot say anything honest about.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DATASET_VERSION = "1.0.0"

LABELS: tuple[str, ...] = ("human", "ai_generated", "ai_polished")
LABEL_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = {i: label for label, i in LABEL_TO_INDEX.items()}

# Where a sample came from. Kept explicit so a bootstrap-only corpus can never be
# mistaken for a corpus of real human and real machine writing.
SOURCES: tuple[str, ...] = (
    "seed_authored",  # hand-written seed essays shipped with the repo
    "seed_variant",  # seed + human-style editing noise
    "ingested_real",  # real essays the operator added to data/raw/human/
    "groq",  # generated or edited by a real hosted model
    "bootstrap_procedural",  # offline template generator (AI_GENERATED proxy)
    "bootstrap_rule_polish",  # offline rule-based polisher (AI_POLISHED proxy)
)


@dataclass(slots=True)
class Sample:
    record_id: str
    label: str
    text: str
    group_id: str
    """Leakage key. Every sample derived from the same underlying essay shares a
    group id, and splits are made over groups — never over samples."""
    source: str
    topic: str = "unspecified"
    length_band: str = "medium"
    voice: str = "unspecified"
    l2_english: bool = False
    model: str | None = None
    strategy: str | None = None
    """Generation strategy (AI_GENERATED) or polish transform (AI_POLISHED)."""
    temperature: float | None = None
    top_p: float | None = None
    parent_id: str | None = None
    """For AI_POLISHED: the human sample this was derived from."""
    license: str = "synthetic-mit"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    notes: str = ""
    split: str | None = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())

    @property
    def n_chars(self) -> int:
        return len(self.text)

    @property
    def n_paragraphs(self) -> int:
        return len([p for p in self.text.split("\n\n") if p.strip()])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_words"] = self.n_words
        data["n_chars"] = self.n_chars
        data["n_paragraphs"] = self.n_paragraphs
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        known = {f for f in cls.__slots__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def write_jsonl(path: Path, samples: Iterable[Sample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[Sample]:
    if not Path(path).exists():
        return iter(())

    def _iterate() -> Iterator[Sample]:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield Sample.from_dict(json.loads(line))

    return _iterate()


def load_samples(path: Path) -> list[Sample]:
    return list(read_jsonl(path))


def dataset_paths(data_dir: Path) -> dict[str, Path]:
    """Canonical on-disk layout."""
    return {
        "human": data_dir / "human" / "human.jsonl",
        "ai_generated": data_dir / "ai_generated" / "ai_generated.jsonl",
        "ai_polished": data_dir / "ai_polished" / "ai_polished.jsonl",
        "combined": data_dir / "processed" / "corpus.jsonl",
        "splits": data_dir / "processed" / "splits.json",
        "manifest": data_dir / "processed" / "manifest.json",
        "features": data_dir / "processed" / "features.npz",
        "feature_manifest": data_dir / "processed" / "features_manifest.json",
        "pairs": data_dir / "processed" / "polish_pairs.jsonl",
    }
