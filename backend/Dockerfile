# Backend image: FastAPI + the detection pipeline.
#
# Dependencies are installed in their own layer before the source is copied, so
# editing application code does not re-download torch.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_DISABLE_TELEMETRY=1

# curl is needed to install uv; libgomp1 is required by LightGBM at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# --- dependency layer -------------------------------------------------------
COPY pyproject.toml uv.lock README.md ./
# The project itself is not installed yet (no source copied), so skip it.
RUN uv sync --frozen --no-install-project

# --- application layer ------------------------------------------------------
COPY app ./app
COPY ml ./ml
COPY tests ./tests
RUN uv sync --frozen

# Pre-download the language model into the image so a cold container does not
# have to reach the network on its first request. Best-effort: a build without
# network access still produces a working image that downloads on first use.
RUN uv run python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer;\
AutoTokenizer.from_pretrained('distilgpt2');\
AutoModelForCausalLM.from_pretrained('distilgpt2')" || \
    echo "distilgpt2 not pre-cached; it will download on first use"

RUN mkdir -p logs data ml/artifacts

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health/live')"

CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
