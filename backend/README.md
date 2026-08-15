# Backend — AI Essay Detector

FastAPI service + hybrid ML/NLP detection pipeline.
See the [root README](../README.md) for full setup, and [docs/](../docs/) for methodology.

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
