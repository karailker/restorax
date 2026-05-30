from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class JobCreateRequest(BaseModel):
    pipeline_id: str = Field(..., description="Pipeline preset ID or YAML filename (without .yaml)")
    output_format: str = Field("mp4", description="Output container format")
    output_codec: str = Field("libx264", description="Video codec")
    output_crf: int = Field(18, ge=0, le=51, description="CRF quality factor (0=lossless, 51=worst)")
    preserve_audio: bool = Field(True, description="Copy audio stream from input")


class JobResponse(BaseModel):
    id: str
    status: str
    progress: float
    pipeline_id: str
    input_path: str
    output_path: str | None
    error: str | None
    metrics: dict
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None
    celery_task_id: str | None

    model_config = {"from_attributes": True}


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class BranchInfo(BaseModel):
    branch_index: int
    name: str
    status: str
    progress: float
    output_path: str | None = None


class BranchListResponse(BaseModel):
    job_id: str
    branches: list[BranchInfo]


class MergeRequest(BaseModel):
    strategy: str = Field(..., description="'blend' or 'select'")
    branch_index: int = Field(0, description="Used when strategy='select'")
