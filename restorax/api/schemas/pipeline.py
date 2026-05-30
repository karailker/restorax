from datetime import datetime

from pydantic import BaseModel, Field


class PipelineCreateRequest(BaseModel):
    id: str = Field(..., description="Unique pipeline ID slug (e.g. 'my_pipeline')")
    name: str = Field(..., description="Human-readable name")
    description: str = Field("", description="Optional description")
    config: dict = Field(..., description="Pipeline config dict (same schema as YAML presets)")


class PipelineResponse(BaseModel):
    id: str
    name: str
    description: str
    config: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PipelineListResponse(BaseModel):
    pipelines: list[PipelineResponse]


class DAGCreateRequest(BaseModel):
    id: str = Field(..., description="Unique DAG ID slug (e.g. 'film_restoration_dag')")
    name: str
    description: str = ""
    config: dict = Field(..., description="Serialised DAG dict from DAGSerializer.to_dict()")


class DAGResponse(BaseModel):
    id: str
    name: str
    description: str
    config: dict
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
