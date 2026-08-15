"""Shared pytest fixtures.

Two important decisions:

* The heavy pipeline (spaCy + distilgpt2) is loaded **once per session** via the
  ``pipeline`` fixture, because loading costs ~15 s and there is no value in
  paying it per test.
* Tests that need MongoDB are marked ``integration`` and skip cleanly when it is
  not running, so ``uv run pytest`` passes on a machine with no database.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("SAVE_ESSAYS", "false")
os.environ.setdefault("REDIS_ENABLED", "false")
os.environ.setdefault("KAFKA_ENABLED", "false")
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_LEVEL", "WARNING")

BACKEND_ROOT = Path(__file__).resolve().parent.parent

HUMAN_LIKE_ESSAY = """The robot never worked. That is the honest summary of my sophomore year. I spent seven months on a line-following car that could not follow a line, and I want to explain why that matters to me.

My design was bad from the start. I used two cheap IR sensors mounted too close together, maybe four centimetres apart, because that was what fit on the breadboard I already owned. When the car hit a curve the sensors both read the same value and the controller just guessed. I didn't know the phrase "insufficient sensor separation" then.

I rebuilt it. Not immediately, though. I put the whole thing in a shoebox under my bed for about six weeks and told my mom I was done with robotics. Then in January I got bored and pulled it out again, and this time I actually read the sensor datasheet instead of guessing.

The car placed fourth in April. Not first. Fourth. But it finished the course three times out of three, and I still chase that feeling."""

MACHINE_LIKE_ESSAY = """From an early age, I have been drawn to robotics. What began as a modest interest gradually developed into a genuine commitment. Moreover, the environment in which I worked was demanding, and it required consistent effort. Furthermore, I approached the work methodically, building my understanding one step at a time.

The most significant challenge arose when my initial approach proved inadequate. Additionally, progress was neither linear nor guaranteed, and there were periods of real difficulty. Consequently, I encountered a setback that forced me to reconsider my fundamental assumptions.

The turning point came when I decided to rebuild my approach from first principles. Moreover, recognising the limits of my method, I sought guidance and revised my strategy. It required patience, precision, and a willingness to fail.

The experience instilled in me a deeper appreciation for patience and iteration. Ultimately, this process cultivated in me a durable capacity for intellectual humility."""


@pytest.fixture(scope="session")
def human_essay() -> str:
    return HUMAN_LIKE_ESSAY


@pytest.fixture(scope="session")
def machine_essay() -> str:
    return MACHINE_LIKE_ESSAY


@pytest.fixture(scope="session")
def settings():  # noqa: ANN201
    from app.config import get_settings

    return get_settings()


@pytest.fixture(scope="session")
def pipeline():  # noqa: ANN201
    """A warmed feature extractor. Session-scoped: model loading dominates cost."""
    from app.services.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    extractor.warmup()
    return extractor


@pytest.fixture(scope="session")
def parsed_human(pipeline, human_essay):  # noqa: ANN001, ANN201
    return pipeline.extract(human_essay)


@pytest.fixture(scope="session")
def parsed_machine(pipeline, machine_essay):  # noqa: ANN001, ANN201
    return pipeline.extract(machine_essay)


@pytest.fixture(scope="session")
def trained_models():  # noqa: ANN201
    """The loaded artifacts, or ``None`` when the model has not been trained."""
    from app.services.classifier import detector_models
    from app.services.detector import detector

    detector.load()
    return detector_models if detector_models.ready else None


@pytest.fixture(scope="session")
def client(trained_models):  # noqa: ANN001, ANN201
    """A FastAPI TestClient with the app's lifespan run."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture
def artifacts_dir() -> Path:
    return BACKEND_ROOT / "ml" / "artifacts"
