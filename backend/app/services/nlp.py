"""spaCy pipeline singleton, paragraph/sentence segmentation.

The pipeline is loaded once per process (model loading dominates cost) and is
shared by the API and the offline ML scripts. If the spaCy model is missing the
module degrades to a regex sentence splitter and records
``segmentation_backend="regex"`` so that the loss of POS/dependency features is
visible rather than silent.
"""

from __future__ import annotations

import re
import threading

from app.config import get_settings
from app.core.logging import get_logger, log_event

logger = get_logger("app.nlp")

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_FALLBACK = re.compile(
    r"""
    (?<=[.!?])          # sentence-final punctuation
    ["')\]]*            # optional closing quotes/brackets
    (?:\s+|\n)          # whitespace
    (?=[A-Z"'(\[]|$)    # next sentence starts with a capital or quote
    """,
    re.VERBOSE,
)
# Abbreviations that must not end a sentence in the fallback splitter.
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Ph\.D|U\.S|a\.m|p\.m)\.$",
    re.IGNORECASE,
)


class NlpPipeline:
    """Thread-safe lazily-loaded spaCy pipeline."""

    def __init__(self) -> None:
        self._nlp = None
        self._lock = threading.Lock()
        self._load_error: str | None = None
        self.model_name = get_settings().spacy_model
        self.load_time_ms: float | None = None

    # ------------------------------------------------------------- loading
    def load(self) -> object | None:
        if self._nlp is not None or self._load_error is not None:
            return self._nlp
        with self._lock:
            if self._nlp is not None or self._load_error is not None:
                return self._nlp
            import time

            started = time.perf_counter()
            try:
                import spacy

                # The tagger/parser/attribute_ruler are all needed (POS +
                # dependency features); NER and lemmatizer are not, and dropping
                # them roughly halves parse time.
                self._nlp = spacy.load(
                    self.model_name, exclude=["ner", "lemmatizer", "textcat"]
                )
                self._nlp.max_length = 2_000_000
                self.load_time_ms = round((time.perf_counter() - started) * 1000, 2)
                log_event(
                    logger,
                    "spacy.loaded",
                    model=self.model_name,
                    load_ms=self.load_time_ms,
                    components=",".join(self._nlp.pipe_names),
                )
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                log_event(
                    logger,
                    "spacy.load_failed",
                    level="warning",
                    model=self.model_name,
                    type=type(exc).__name__,
                    hint="falling back to regex segmentation; syntactic features will be limited",
                )
        return self._nlp

    @property
    def available(self) -> bool:
        return self.load() is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def info(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "loaded": self._nlp is not None,
            "load_time_ms": self.load_time_ms,
            "error": self._load_error,
            "backend": "spacy" if self._nlp is not None else "regex",
        }

    # ------------------------------------------------------------- parsing
    def parse(self, text: str):  # noqa: ANN201 - spaCy Doc or None
        nlp = self.load()
        if nlp is None:
            return None
        return nlp(text)


nlp_pipeline = NlpPipeline()


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #
def find_paragraph_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of paragraphs, split on blank lines.

    Single newlines are treated as soft wraps (people paste essays with hard
    wrapping), blank lines as real paragraph breaks. If there are no blank lines
    at all we fall back to single-newline splitting so that a hand-typed essay
    still produces a sensible paragraph structure.
    """
    if not text.strip():
        return []

    spans: list[tuple[int, int]] = []
    cursor = 0
    for chunk in _PARAGRAPH_SPLIT.split(text):
        if not chunk.strip():
            cursor = text.find(chunk, cursor) + len(chunk)
            continue
        start = text.find(chunk, cursor)
        if start < 0:  # pragma: no cover - defensive
            start = cursor
        end = start + len(chunk)
        spans.append((start, end))
        cursor = end

    if len(spans) <= 1 and "\n" in text.strip():
        spans = []
        cursor = 0
        for line in text.split("\n"):
            if line.strip():
                start = text.find(line, cursor)
                spans.append((start, start + len(line)))
                cursor = start + len(line)
            else:
                cursor += len(line) + 1
    return spans or [(0, len(text))]


def _fallback_sentence_spans(text: str, offset: int = 0) -> list[tuple[int, int]]:
    """Regex sentence spans, used when spaCy is unavailable."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_FALLBACK.finditer(text):
        end = match.start()
        candidate = text[start:end].strip()
        if candidate and not _ABBREV.search(candidate):
            spans.append((offset + start, offset + end))
            start = match.end()
    tail = text[start:].strip()
    if tail:
        # Recover exact offsets for the tail (strip() moved them).
        lead = len(text[start:]) - len(text[start:].lstrip())
        spans.append((offset + start + lead, offset + len(text.rstrip())))
    return [(s, e) for s, e in spans if e > s]


def segment(text: str):  # noqa: ANN201 - returns ParsedDocument
    """Segment and parse ``text`` into a :class:`ParsedDocument`."""
    from app.services.document import ParagraphUnit, ParsedDocument, SentenceUnit

    paragraph_spans = find_paragraph_spans(text)
    doc = nlp_pipeline.parse(text)
    backend = "spacy" if doc is not None else "regex"

    sentences: list[SentenceUnit] = []
    if doc is not None:
        raw = [
            (sent.start_char, sent.end_char, sent)
            for sent in doc.sents
            if sent.text.strip()
        ]
    else:
        raw = [(s, e, None) for s, e in _fallback_sentence_spans(text)]

    for start, end, span in raw:
        # Trim leading/trailing whitespace out of the span while keeping offsets
        # aligned with the source text (the frontend slices the same string).
        surface = text[start:end]
        lead = len(surface) - len(surface.lstrip())
        trail = len(surface) - len(surface.rstrip())
        s, e = start + lead, end - trail
        if e <= s:
            continue
        para_index = _paragraph_of(s, paragraph_spans)
        sentences.append(
            SentenceUnit(
                index=len(sentences),
                paragraph_index=para_index,
                start=s,
                end=e,
                text=text[s:e],
                span=span,
            )
        )

    paragraphs = [
        ParagraphUnit(index=i, start=s, end=e, text=text[s:e])
        for i, (s, e) in enumerate(paragraph_spans)
    ]
    for sentence in sentences:
        if 0 <= sentence.paragraph_index < len(paragraphs):
            paragraphs[sentence.paragraph_index].sentence_indices.append(sentence.index)

    # Drop paragraphs that ended up with no sentences (e.g. a stray heading of
    # punctuation) but keep indices stable by re-mapping.
    kept = [p for p in paragraphs if p.sentence_indices]
    if len(kept) != len(paragraphs):
        remap = {p.index: i for i, p in enumerate(kept)}
        for i, para in enumerate(kept):
            para.index = i
        for sentence in sentences:
            sentence.paragraph_index = remap.get(sentence.paragraph_index, 0)
        paragraphs = kept

    return ParsedDocument(
        text=text,
        sentences=sentences,
        paragraphs=paragraphs,
        doc=doc,
        segmentation_backend=backend,
    )


def _paragraph_of(position: int, spans: list[tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(spans):
        if start <= position < end:
            return i
    # Position falls in the gap between paragraphs - attach to the nearest
    # preceding paragraph.
    for i in range(len(spans) - 1, -1, -1):
        if spans[i][0] <= position:
            return i
    return 0
