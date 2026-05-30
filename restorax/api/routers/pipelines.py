"""
Pipeline template CRUD endpoints.

POST   /pipelines          — create pipeline template
GET    /pipelines          — list all templates
GET    /pipelines/{id}     — get template
PUT    /pipelines/{id}     — update template
DELETE /pipelines/{id}     — delete template
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.ext.asyncio import AsyncSession

from restorax.api.deps import get_db
from restorax.api.schemas.pipeline import (
    DAGCreateRequest,
    DAGResponse,
    PipelineCreateRequest,
    PipelineListResponse,
    PipelineResponse,
)
from restorax.core.exceptions import DAGValidationError, PipelineConfigError
from restorax.dag.serializer import DAGSerializer
from restorax.db.models import PipelineTemplateModel
from restorax.db.repositories.pipeline_repo import PipelineRepository

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.post("", response_model=PipelineResponse, status_code=status.HTTP_201_CREATED)
async def create_pipeline(
    req: PipelineCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> PipelineResponse:
    repo = PipelineRepository(db)
    try:
        existing = await repo.get(req.id)
        raise HTTPException(status_code=409, detail=f"Pipeline '{req.id}' already exists")
    except PipelineConfigError:
        pass  # does not exist — safe to create

    p = PipelineTemplateModel(
        id=req.id,
        name=req.name,
        description=req.description,
        config=req.config,
    )
    created = await repo.create(p)
    return PipelineResponse.model_validate(created)


@router.get("", response_model=PipelineListResponse)
async def list_pipelines(db: AsyncSession = Depends(get_db)) -> PipelineListResponse:
    repo = PipelineRepository(db)
    pipelines = await repo.list_all()
    return PipelineListResponse(pipelines=[PipelineResponse.model_validate(p) for p in pipelines])


@router.get("/{pipeline_id}", response_model=PipelineResponse)
async def get_pipeline(pipeline_id: str, db: AsyncSession = Depends(get_db)) -> PipelineResponse:
    repo = PipelineRepository(db)
    try:
        p = await repo.get(pipeline_id)
    except PipelineConfigError:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return PipelineResponse.model_validate(p)


@router.put("/{pipeline_id}", response_model=PipelineResponse)
async def update_pipeline(
    pipeline_id: str,
    req: PipelineCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> PipelineResponse:
    repo = PipelineRepository(db)
    try:
        p = await repo.update(pipeline_id, name=req.name, description=req.description, config=req.config)
    except PipelineConfigError:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return PipelineResponse.model_validate(p)


@router.delete("/{pipeline_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_pipeline(pipeline_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    repo = PipelineRepository(db)
    try:
        await repo.delete(pipeline_id)
    except PipelineConfigError:
        raise HTTPException(status_code=404, detail=f"Pipeline '{pipeline_id}' not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ── DAG endpoints ─────────────────────────────────────────────────────────────


@router.post("/dag", response_model=DAGResponse, status_code=status.HTTP_201_CREATED, tags=["dag"])
async def create_dag(
    req: DAGCreateRequest,
    db: AsyncSession = Depends(get_db),
) -> DAGResponse:
    """Create a DAG pipeline. Config must be a valid DAGSerializer.to_dict() output."""
    # Validate the DAG structure before persisting
    try:
        DAGSerializer.from_dict(req.config)
    except (DAGValidationError, Exception) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid DAG config: {exc}")

    repo = PipelineRepository(db)
    p = PipelineTemplateModel(
        id=req.id,
        name=req.name,
        description=req.description,
        config=req.config,  # already has schema_type: "dag"
    )
    try:
        created = await repo.create(p)
    except Exception:
        raise HTTPException(status_code=409, detail=f"DAG '{req.id}' already exists")
    return DAGResponse.model_validate(created)


@router.get("/dag/{dag_id}", response_model=DAGResponse, tags=["dag"])
async def get_dag(dag_id: str, db: AsyncSession = Depends(get_db)) -> DAGResponse:
    repo = PipelineRepository(db)
    try:
        p = await repo.get(dag_id)
    except PipelineConfigError:
        raise HTTPException(status_code=404, detail=f"DAG '{dag_id}' not found")
    return DAGResponse.model_validate(p)
