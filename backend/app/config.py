"""Application configuration.

All runtime behaviour is driven by environment variables (or a local ``.env``
file). Nothing that resembles a credential has a usable default: the defaults
here are development-only endpoints on localhost.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_ROOT / "data"
ARTIFACTS_DIR = BACKEND_ROOT / "ml" / "artifacts"


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),  # we legitimately use `model_*` names
    )

    # ------------------------------------------------------------------ app
    app_env: str = "development"
    app_name: str = "ai-essay-detector"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"
    log_dir: str = "logs"
    log_json: bool = False
    debug_timings: bool = True

    # ------------------------------------------------------------- security
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_essay_chars: int = 60_000
    min_essay_chars: int = 200
    max_request_bytes: int = 1_000_000
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60

    # -------------------------------------------------------------- privacy
    save_essays: bool = False
    """When False the raw essay body is never persisted — only metadata,
    features and per-sentence offsets/scores. Sentence text is stored only when
    this flag is on."""
    analysis_retention_days: int = 30

    # ------------------------------------------------------------- mongodb
    mongodb_url: str = "mongodb://localhost:27017"
    mongodb_database: str = "ai_essay_detector"
    mongodb_enabled: bool = True
    mongodb_timeout_ms: int = 3000

    # --------------------------------------------------------------- redis
    redis_enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl_seconds: int = 86_400

    # --------------------------------------------------------------- kafka
    kafka_enabled: bool = False
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_analysis_topic: str = "essay.analysis.requests"
    kafka_result_topic: str = "essay.analysis.results"
    kafka_consumer_group: str = "essay-analysis-workers"
    async_threshold_chars: int = 25_000
    """Essays longer than this are routed through Kafka when it is enabled."""

    # ------------------------------------------------------- language model
    lm_model_name: str = "distilgpt2"
    lm_device: str = "cpu"
    lm_max_window: int = 512
    lm_stride: int = 384
    lm_lazy_load: bool = False
    spacy_model: str = "en_core_web_sm"

    # ------------------------------------------------------------ artifacts
    model_dir: str = str(ARTIFACTS_DIR)
    detector_version: str = "1.0.0"

    # ------------------------------------------------ dataset generation (Groq)
    groq_api_key: str | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_models: str = (
        "llama-3.3-70b-versatile,llama-3.1-8b-instant,openai/gpt-oss-20b,gemma2-9b-it"
    )
    groq_request_timeout: int = 120

    # ------------------------------------------------------------- helpers
    @field_validator("log_level")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def groq_model_list(self) -> list[str]:
        return [m.strip() for m in self.groq_models.split(",") if m.strip()]

    @property
    def artifacts_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def data_path(self) -> Path:
        return DATA_DIR

    @property
    def log_path(self) -> Path:
        p = Path(self.log_dir)
        return p if p.is_absolute() else BACKEND_ROOT / p

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that patch the environment."""
    get_settings.cache_clear()


# Convenience for module-level use in non-request code (ML scripts).
settings = get_settings()

# Silence HF telemetry / tokenizer fork warnings for the whole process.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
