"""Structured representation of an essay.

Everything downstream works against these objects rather than raw strings, so
character offsets stay consistent between the frontend highlighter, the LM token
alignment and the persisted per-sentence rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    pass


@dataclass(slots=True)
class SentenceUnit:
    """One analysed sentence."""

    index: int
    """0-based position in the document."""
    paragraph_index: int
    start: int
    """Character offset into the normalised essay text (inclusive)."""
    end: int
    """Character offset (exclusive)."""
    text: str
    span: Any | None = field(default=None, repr=False)
    """The spaCy ``Span``, when the spaCy pipeline is available."""

    # Filled in by the measurement layers.
    lm: dict[str, float] = field(default_factory=dict, repr=False)
    stylometry: dict[str, float] = field(default_factory=dict, repr=False)
    syntax: dict[str, float] = field(default_factory=dict, repr=False)
    context: dict[str, float] = field(default_factory=dict, repr=False)
    corpus: dict[str, float] = field(default_factory=dict, repr=False)
    pos_distribution: dict[str, float] = field(default_factory=dict, repr=False)
    function_word_profile: dict[str, float] = field(default_factory=dict, repr=False)
    tokens: list[str] = field(default_factory=list, repr=False)
    content_words: list[str] = field(default_factory=list, repr=False)
    pos_sequence: list[str] = field(default_factory=list, repr=False)

    @property
    def n_words(self) -> int:
        return int(self.stylometry.get("sty_n_words", len(self.text.split())))

    def features(self) -> dict[str, float]:
        """The flat sentence-level feature vector."""
        merged: dict[str, float] = {}
        for group in (self.lm, self.stylometry, self.syntax, self.context, self.corpus):
            merged.update(group)
        return merged


@dataclass(slots=True)
class ParagraphUnit:
    index: int
    start: int
    end: int
    text: str
    sentence_indices: list[int] = field(default_factory=list)

    @property
    def n_sentences(self) -> int:
        return len(self.sentence_indices)


@dataclass(slots=True)
class ParsedDocument:
    """An essay after normalisation, segmentation and parsing."""

    text: str
    sentences: list[SentenceUnit]
    paragraphs: list[ParagraphUnit]
    doc: Any | None = field(default=None, repr=False)
    """The spaCy ``Doc`` for the whole essay, when available."""
    segmentation_backend: str = "spacy"
    """``spacy`` or ``regex`` - surfaced in debug info so a degraded parse is
    never silently mistaken for a full one."""

    @property
    def n_words(self) -> int:
        return sum(s.n_words for s in self.sentences)

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def n_paragraphs(self) -> int:
        return len(self.paragraphs)

    def sentences_in(self, paragraph_index: int) -> list[SentenceUnit]:
        para = self.paragraphs[paragraph_index]
        return [self.sentences[i] for i in para.sentence_indices]

    def sentence_lengths(self) -> list[int]:
        return [s.n_words for s in self.sentences]
