<p align="center">
  <a href="../README.md">Home</a>
  &middot;
  <a href="../docs/architecture.md">Architecture</a>
  &middot;
  <a href="../docs/detection-methodology.md">Methodology</a>
  &middot;
  <a href="../docs/dataset.md">Dataset</a>
  &middot;
  <a href="../docs/evaluation.md">Evaluation</a>
  &middot;
  <a href="../docs/api.md">API</a>
  &middot;
  <a href="../docs/privacy.md">Privacy</a>
  &middot;
  <a href="../docs/limitations.md">Limitations</a>
  &middot;
  <a href="../docs/README.md">All docs</a>
</p>

---

# Backend - AI Essay Detector

FastAPI service + hybrid ML/NLP detection pipeline.

Dataset operations: [`data/README.md`](data/README.md) ·
[`data/raw/README.md`](data/raw/README.md)

## Quick start (Windows PowerShell)

```powershell
uv venv
uv sync
cp .env.example .env
uv run python -m ml.training.prepare_dataset
uv run python -m ml.training.extract_features
uv run python -m ml.training.train
uv run python -m ml.evaluation.evaluate
uv run python -m ml.evaluation.find_failures
uv run uvicorn app.main:app --reload --port 8000
```
