"""Structured logging built on `logifyx`.

Every module obtains its logger through :func:`get_logger`. Two rules are
enforced by construction rather than by convention:

1. **No essay text ever reaches a log record.** :func:`safe_text_meta` is the
   only sanctioned way to describe a piece of user text in a log, and it emits
   lengths and a truncated SHA-256 digest - never characters.
2. **No `print()`.** Anything worth showing goes through logifyx so it lands in
   the console *and* the rotating file handler with consistent structure.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from logifyx import ContextLoggerAdapter, Logifyx, setup_logify

from app.config import get_settings

_CONFIGURED = False
_LOGGERS: dict[str, Logifyx] = {}

# ``setup_logify()`` installs Logifyx as the *global* logger class, which means
# every third-party library starts emitting through our console handler. Torch,
# transformers and httpx are extremely chatty at INFO (a model load prints the
# entire config), so they are pinned to WARNING. Our own ``app.*`` loggers are
# unaffected - they are created explicitly with the configured level.
_NOISY_LIBRARIES: tuple[str, ...] = (
    "aiokafka",
    "asyncio",
    "filelock",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "matplotlib",
    "motor",
    "numba",
    "pymongo",
    "sklearn",
    "spacy",
    "thinc",
    "torch",
    "transformers",
    "urllib3",
    "uvicorn.access",
)

# Patterns that must never survive into a log line even by accident.
_PII_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<email>"),
    (re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)"), "<phone>"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<ssn>"),
)


def configure_logging() -> None:
    """Register logifyx as the global logger class. Safe to call repeatedly."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    settings.log_path.mkdir(parents=True, exist_ok=True)
    setup_logify()
    _quiet_third_party()
    _CONFIGURED = True


def _quiet_third_party() -> None:
    """Pin noisy library loggers to WARNING.

    Called once at startup and again right after torch/transformers are imported:
    those packages configure their own child loggers at import time with explicit
    levels, which override anything set on the parent beforehand.
    """
    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Sweep already-created child loggers (e.g. ``torch._dynamo.eval_frame``),
    # whose explicit levels would otherwise win over the parent's.
    for name in list(logging.root.manager.loggerDict):
        if name.split(".")[0] in _NOISY_LIBRARIES:
            existing = logging.getLogger(name)
            if existing.level and existing.level < logging.WARNING:
                existing.setLevel(logging.WARNING)
    try:  # transformers has its own verbosity switch on top of `logging`
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:  # pragma: no cover - transformers is optional at import time
        pass


def quiet_third_party() -> None:
    """Public re-entry point, used after heavy imports."""
    _quiet_third_party()


def get_logger(name: str, *, file: str | None = None) -> Logifyx:
    """Return a per-name singleton logifyx logger."""
    configure_logging()
    if name in _LOGGERS:
        return _LOGGERS[name]
    settings = get_settings()
    logger = Logifyx(
        name=name,
        level=getattr(logging, settings.log_level, logging.INFO),
        file=file or f"{settings.app_name}.log",
        log_dir=str(settings.log_path),
        color=not settings.log_json,
        json_mode=settings.log_json,
        mask=True,
    )
    logger.propagate = False
    _LOGGERS[name] = logger
    return logger


def set_logger_level(name: str, level: int | str) -> None:
    """Change the level of one of *our* loggers.

    ``logging.getLogger(name)`` is not sufficient: :class:`Logifyx` instances are
    constructed directly rather than through the manager, so the manager may hold
    a different object under the same name. Batch scripts use this to silence the
    per-document pipeline chatter without touching anything else.
    """
    resolved = getattr(logging, level, level) if isinstance(level, str) else level
    if name in _LOGGERS:
        _LOGGERS[name].setLevel(resolved)
        for handler in _LOGGERS[name].handlers:
            handler.setLevel(resolved)
    logging.getLogger(name).setLevel(resolved)


def with_context(logger: Logifyx, **context: Any) -> ContextLoggerAdapter:
    """Attach request-scoped context (request_id, analysis_id, ...) to a logger."""
    return ContextLoggerAdapter(logger, context)


# --------------------------------------------------------------------------- #
# Privacy helpers
# --------------------------------------------------------------------------- #
def text_digest(text: str, length: int = 16) -> str:
    """Stable, non-reversible identifier for a piece of text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def content_hash(text: str, *, detector_version: str, model_version: str) -> str:
    """Cache key: SHA256(essay + detector_version + model_version)."""
    payload = f"{text}\x00{detector_version}\x00{model_version}".encode()
    return hashlib.sha256(payload).hexdigest()


def safe_text_meta(text: str, *, prefix: str = "text") -> dict[str, Any]:
    """Describe user text without revealing it.

    >>> sorted(safe_text_meta("hello world").keys())
    ['text_chars', 'text_digest', 'text_words']
    """
    return {
        f"{prefix}_chars": len(text),
        f"{prefix}_words": len(text.split()),
        f"{prefix}_digest": text_digest(text),
    }


def scrub(message: str) -> str:
    """Defensive scrub of obvious PII from an arbitrary string (e.g. an
    exception message that may echo user input)."""
    out = message
    for pattern, replacement in _PII_PATTERNS:
        out = pattern.sub(replacement, out)
    return out


@contextmanager
def log_duration(
    logger: Logifyx | ContextLoggerAdapter,
    event: str,
    /,
    **fields: Any,
) -> Iterator[dict[str, Any]]:
    """Time a block and emit one structured record when it finishes.

    The yielded dict can be mutated to add fields discovered during the block;
    it also carries ``duration_ms`` after the block exits.
    """
    extra: dict[str, Any] = dict(fields)
    started = time.perf_counter()
    try:
        yield extra
    except Exception as exc:  # pragma: no cover - re-raised immediately
        extra["duration_ms"] = round((time.perf_counter() - started) * 1000, 2)
        extra["error"] = type(exc).__name__
        logger.error(f"{event}.failed | {_fmt(extra)}")
        raise
    duration_ms = round((time.perf_counter() - started) * 1000, 2)
    extra["duration_ms"] = duration_ms
    logger.info(f"{event} | {_fmt(extra)}")


def _fmt(fields: dict[str, Any]) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items())


def log_event(
    logger: Logifyx | ContextLoggerAdapter,
    event: str,
    /,
    level: str = "info",
    **fields: Any,
) -> None:
    """Emit a single structured event line."""
    getattr(logger, level)(f"{event} | {_fmt(fields)}" if fields else event)
