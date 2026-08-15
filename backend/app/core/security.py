"""Input validation and request hardening helpers."""

from __future__ import annotations

import re
import unicodedata

from app.config import Settings
from app.core.exceptions import EmptyEssayError, EssayTooLongError, EssayTooShortError

# Zero-width / bidi / private-use characters. These are sometimes used to try to
# confuse tokenisers, and they carry no meaning in an essay.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_TRAILING_WS = re.compile(r"[ \t]+(?=\n)")
_MANY_SPACES = re.compile(r"[ \t]{2,}")


def normalise_essay(text: str) -> str:
    """Canonicalise whitespace and unicode without altering wording.

    Normalisation is deliberately conservative - the detector measures *style*,
    so we must not "fix" the author's punctuation or spacing patterns beyond
    removing artefacts of copy-pasting.
    """
    out = unicodedata.normalize("NFKC", text)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _INVISIBLE.sub("", out)
    out = _CONTROL.sub("", out)
    out = _TRAILING_WS.sub("", out)
    out = _MANY_SPACES.sub(" ", out)
    out = _MANY_NEWLINES.sub("\n\n", out)
    return out.strip()


def validate_essay(text: str | None, settings: Settings) -> str:
    """Normalise and bounds-check an incoming essay.

    Raises :class:`EmptyEssayError`, :class:`EssayTooShortError` or
    :class:`EssayTooLongError`.
    """
    if text is None or not text.strip():
        raise EmptyEssayError()

    cleaned = normalise_essay(text)
    if not cleaned:
        raise EmptyEssayError()

    if len(cleaned) > settings.max_essay_chars:
        raise EssayTooLongError(
            f"The essay is {len(cleaned):,} characters; the maximum is "
            f"{settings.max_essay_chars:,}.",
            length=len(cleaned),
            limit=settings.max_essay_chars,
        )
    if len(cleaned) < settings.min_essay_chars:
        raise EssayTooShortError(
            f"The essay is {len(cleaned):,} characters; at least "
            f"{settings.min_essay_chars:,} are needed for a meaningful analysis.",
            length=len(cleaned),
            minimum=settings.min_essay_chars,
        )
    return cleaned
