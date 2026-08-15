"""Evaluation endpoints backing the research dashboard."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.api.dependencies import DbDep
from app.models.evaluation import load_evaluation_bundle
from app.schemas.analysis import EvaluationResponse

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


@router.get(
    "",
    response_model=EvaluationResponse,
    summary="Evaluation report, failure analysis and dataset card",
    description=(
        "Serves the artifacts produced by `ml.evaluation.evaluate` and "
        "`ml.evaluation.find_failures`. Returns `available: false` with a message "
        "rather than fabricating numbers when the pipeline has not been run."
    ),
)
async def get_evaluation():  # noqa: ANN201
    return load_evaluation_bundle()


@router.get(
    "/runs",
    summary="Stored evaluation runs",
    description="Historical evaluation runs from MongoDB, newest first.",
)
async def list_runs(db: DbDep, limit: int = Query(default=5, ge=1, le=25)):  # noqa: ANN201
    if not db.available:
        return {"available": False, "runs": [], "message": "MongoDB is unavailable."}
    latest = await db.latest_evaluation_run()
    return {
        "available": latest is not None,
        "latest": latest,
        "message": None
        if latest
        else (
            "No stored runs. Run `uv run python -m ml.evaluation.evaluate "
            "--store-in-mongo`."
        ),
    }


@router.get(
    "/model-versions",
    summary="Registered model versions",
)
async def model_versions(db: DbDep, limit: int = Query(default=10, ge=1, le=50)):  # noqa: ANN201
    if not db.available:
        return {"available": False, "versions": []}
    versions = await db.list_model_versions(limit=limit)
    return {"available": True, "versions": versions}
