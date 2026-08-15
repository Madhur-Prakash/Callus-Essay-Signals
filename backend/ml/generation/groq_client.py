"""Minimal Groq chat-completions client used for dataset generation only.

Scope discipline matters here: this client is used **exclusively offline, to
build training data**. It is never imported by the API, and no request path in
``app/`` can reach a hosted model. The detector's verdict is produced entirely by
our own trained classifier - see ``docs/detection-methodology.md``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import get_settings
from app.core.logging import get_logger, log_event

logger = get_logger("ml.groq")


class GroqUnavailableError(RuntimeError):
    """Raised when no API key is configured."""


@dataclass(slots=True)
class Completion:
    text: str
    model: str
    temperature: float
    top_p: float
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: int | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key or settings.groq_api_key
        self.base_url = (base_url or settings.groq_base_url).rstrip("/")
        self.timeout = timeout or settings.groq_request_timeout
        self.models = settings.groq_model_list

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 1200,
        max_retries: int = 4,
    ) -> Completion:
        if not self.api_key:
            raise GroqUnavailableError(
                "GROQ_API_KEY is not set. Use the offline bootstrap generator instead "
                "(`uv run python -m ml.generation.bootstrap_corpus`)."
            )

        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(max_retries):
            started = time.perf_counter()
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        f"{self.base_url}/chat/completions", json=payload, headers=headers
                    )
                if response.status_code == 429:
                    wait = min(60.0, 2.0 * (2**attempt))
                    log_event(
                        logger,
                        "groq.rate_limited",
                        level="warning",
                        model=model,
                        attempt=attempt + 1,
                        sleep_s=wait,
                    )
                    time.sleep(wait)
                    continue
                response.raise_for_status()
                body = response.json()
                usage = body.get("usage") or {}
                return Completion(
                    text=body["choices"][0]["message"]["content"].strip(),
                    model=model,
                    temperature=temperature,
                    top_p=top_p,
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except Exception as exc:  # noqa: BLE001 - retried below
                last_error = exc
                log_event(
                    logger,
                    "groq.request_failed",
                    level="warning",
                    model=model,
                    attempt=attempt + 1,
                    type=type(exc).__name__,
                )
                time.sleep(min(30.0, 1.5 * (2**attempt)))

        raise RuntimeError(f"Groq request failed after {max_retries} attempts") from last_error
