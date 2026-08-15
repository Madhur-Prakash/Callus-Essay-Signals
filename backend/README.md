# Backend — AI Essay Detector

FastAPI service + hybrid ML/NLP detection pipeline.

[Home](../README.md) · [Docs index](../docs/README.md) ·
[Architecture](../docs/architecture.md) ·
[Methodology](../docs/detection-methodology.md) · [Dataset](../docs/dataset.md) ·
[Evaluation](../docs/evaluation.md) · [API](../docs/api.md) ·
[Privacy](../docs/privacy.md) · [Limitations](../docs/limitations.md)

Dataset operations: [`data/README.md`](data/README.md) ·
[`data/raw/README.md`](data/raw/README.md)

## Quick start (Windows PowerShell)

```powershell
uv venv
uv sync
Copy-Item .env.example .env
uv run python -m ml.training.prepare_dataset
uv run python -m ml.training.extract_features
uv run python -m ml.training.train
uv run python -m ml.evaluation.evaluate
uv run python -m ml.evaluation.find_failures
uv run uvicorn app.main:app --reload --port 8000
```
