"""
Job management endpoints.

POST   /jobs         — submit a new restoration job (file upload)
POST   /jobs/batch   — submit multiple files at once (one job per file)
GET    /jobs         — list jobs
GET    /jobs/{id}    — get job status
GET    /jobs/{id}/download — download output video
DELETE /jobs/{id}    — cancel / delete a job
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from restorax.api.deps import get_db
from restorax.api.schemas.job import (
    BranchInfo,
    BranchListResponse,
    JobListResponse,
    JobResponse,
)
from restorax.config import settings
from restorax.core.exceptions import JobNotFoundError
from restorax.db.models import JobModel
from restorax.db.repositories.job_repo import JobRepository

router = APIRouter(prefix="/jobs", tags=["jobs"])

# Mount point for artifacts under <storage_local_root>/outputs (branch previews).
FILES_URL_PREFIX = "/files"

# Extensions accepted for output files (used in output.{output_format} paths)
_ALLOWED_OUTPUT_FORMATS = frozenset({"mp4", "mkv", "avi", "mov", "webm"})


def _validate_output_format(output_format: str) -> None:
    if output_format not in _ALLOWED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported output_format '{output_format}'. "
            f"Allowed: {sorted(_ALLOWED_OUTPUT_FORMATS)}",
        )


# Chunk size for streaming uploads — bounded RAM regardless of file size.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


async def _save_upload(file: UploadFile, dest: Path) -> None:
    """Stream an upload to `dest`, rejecting anything above settings.max_upload_mb."""
    max_bytes = settings.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with dest.open("wb") as out:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if max_bytes and written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {settings.max_upload_mb} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)  # don't leave a truncated file behind
        raise


@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(
    file: UploadFile = File(..., description="Input video file"),
    pipeline_id: str | None = Form(None),
    dag_id: str | None = Form(None, description="DAG pipeline ID (alternative to pipeline_id)"),
    output_format: str = Form("mp4"),
    output_codec: str = Form("libx264"),
    output_crf: int = Form(18, ge=0, le=51),
    preserve_audio: bool = Form(True),
    restore_audio: bool = Form(False, description="Run audio restoration pipeline"),
    db: AsyncSession = Depends(get_db),
) -> JobResponse:
    """Upload a video and enqueue a restoration job."""
    if pipeline_id is None and dag_id is None:
        raise HTTPException(status_code=422, detail="Either pipeline_id or dag_id is required")

    _validate_output_format(output_format)
    job_id = str(uuid.uuid4())

    # Save uploaded file to local storage root
    storage_root = Path(settings.storage_local_root)
    input_dir = storage_root / "inputs" / job_id
    input_dir.mkdir(parents=True, exist_ok=True)

    safe_filename = Path(file.filename or "input.mp4").name
    input_path = input_dir / safe_filename
    await _save_upload(file, input_path)

    output_path = str(storage_root / "outputs" / job_id / f"output.{output_format}")

    # Resolve preset YAML path (sequential pipelines only)
    preset_path = _resolve_preset(pipeline_id) if pipeline_id else None

    # Persist job to database
    repo = JobRepository(db)
    job_model = JobModel(
        id=job_id,
        status="queued",
        input_path=str(input_path),
        pipeline_id=dag_id or pipeline_id,  # store whichever was provided
        output_format=output_format,
        output_codec=output_codec,
        output_crf=output_crf,
        preserve_audio=preserve_audio,
        output_path=output_path,
    )
    await repo.create(job_model)

    # Dispatch appropriate Celery task
    if dag_id is not None:
        from restorax.tasks.job_tasks import run_dag_job

        task = run_dag_job.apply_async(
            kwargs={
                "job_id": job_id,
                "dag_id": dag_id,
                "input_path": str(input_path),
                "output_path": output_path,
            }
        )
    else:
        from restorax.tasks.job_tasks import run_job

        task = run_job.apply_async(
            kwargs={
                "job_id": job_id,
                "pipeline_preset_path": preset_path,
                "input_path": str(input_path),
                "output_path": output_path,
                "restore_audio": restore_audio,
                "output_codec": output_codec,
                "output_crf": output_crf,
                "preserve_audio": preserve_audio,
            }
        )
    await repo.update_status(job_id, status="queued", celery_task_id=task.id)

    job_model = await repo.get(job_id)
    return _to_response(job_model)


@router.get("", response_model=JobListResponse)
async def list_jobs(
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    repo = JobRepository(db)
    jobs = await repo.list_all(limit=limit, offset=offset)
    return JobListResponse(jobs=[_to_response(j) for j in jobs], total=len(jobs))


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)) -> JobResponse:
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from None
    return _to_response(job)


@router.get("/{job_id}/download")
async def download_output(job_id: str, db: AsyncSession = Depends(get_db)) -> FileResponse:
    """Download the output video for a completed job (local storage only)."""
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from None
    if job.status != "completed" or not job.output_path:
        raise HTTPException(status_code=409, detail="Job output not ready")
    output_path = Path(job.output_path)
    # Containment: only serve files under storage_local_root/outputs
    outputs_root = (Path(settings.storage_local_root) / "outputs").resolve()
    if not output_path.resolve().is_relative_to(outputs_root):
        raise HTTPException(status_code=403, detail="Output path outside storage root")
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")
    return FileResponse(
        path=str(output_path),
        media_type="video/mp4",
        filename=output_path.name,
    )


@router.delete("/{job_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job(job_id: str, db: AsyncSession = Depends(get_db)) -> Response:
    repo = JobRepository(db)
    try:
        await repo.delete(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/batch", response_model=JobListResponse, status_code=status.HTTP_201_CREATED)
async def create_batch_jobs(
    files: list[UploadFile] = File(..., description="Multiple video files"),
    pipeline_id: str = Form(...),
    output_format: str = Form("mp4"),
    output_codec: str = Form("libx264"),
    output_crf: int = Form(18),
    preserve_audio: bool = Form(True),
    db: AsyncSession = Depends(get_db),
) -> JobListResponse:
    """
    Submit multiple video files as separate jobs in one request.

    Each file becomes an independent job dispatched to the next available GPU
    queue via the round-robin router (Phase 5.4). Returns the list of created jobs.
    """
    from restorax.tasks.gpu_router import next_gpu_queue

    _validate_output_format(output_format)
    repo = JobRepository(db)
    preset_path = _resolve_preset(pipeline_id)
    storage_root = Path(settings.storage_local_root)
    created: list[JobModel] = []

    for file in files:
        job_id = str(uuid.uuid4())
        input_dir = storage_root / "inputs" / job_id
        input_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(file.filename or "input.mp4").name
        input_path = input_dir / safe_name
        await _save_upload(file, input_path)
        output_path = str(storage_root / "outputs" / job_id / f"output.{output_format}")

        job_model = JobModel(
            id=job_id,
            status="queued",
            input_path=str(input_path),
            pipeline_id=pipeline_id,
            output_format=output_format,
            output_codec=output_codec,
            output_crf=output_crf,
            preserve_audio=preserve_audio,
            output_path=output_path,
        )
        await repo.create(job_model)

        from restorax.tasks.job_tasks import run_job

        gpu_queue = next_gpu_queue()
        task = run_job.apply_async(
            kwargs={
                "job_id": job_id,
                "pipeline_preset_path": preset_path,
                "input_path": str(input_path),
                "output_path": output_path,
                "output_codec": output_codec,
                "output_crf": output_crf,
                "preserve_audio": preserve_audio,
            },
            queue=gpu_queue,
        )
        await repo.update_status(job_id, status="queued", celery_task_id=task.id)
        created.append(await repo.get(job_id))

    return JobListResponse(jobs=[_to_response(j) for j in created], total=len(created))


@router.get("/{job_id}/branches", response_model=BranchListResponse)
async def get_job_branches(job_id: str, db: AsyncSession = Depends(get_db)) -> BranchListResponse:
    """Per-branch name, status and preview image for a DAG job's parallel node.

    The branch list is what the parallel node actually ran (recorded in
    `dag_run["node_metadata"]`), not an inference from node ids.
    """
    repo = JobRepository(db)
    try:
        job = await repo.get(job_id)
    except JobNotFoundError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found") from None

    dag_run: dict[str, Any] = job.dag_run or {}
    node_states: dict[str, Any] = dag_run.get("node_states") or {}
    node_metadata: dict[str, Any] = dag_run.get("node_metadata") or {}

    branches: list[BranchInfo] = []
    for node_id, md in node_metadata.items():
        state = node_states.get(node_id, "pending")
        for entry in md.get("branches", []):
            branches.append(
                BranchInfo(
                    branch_index=entry["index"],
                    name=entry.get("name", ""),
                    node_id=node_id,
                    status=state,
                    progress=1.0 if state == "succeeded" else 0.0,
                    preview_url=_preview_url(entry.get("preview_path")),
                )
            )

    return BranchListResponse(job_id=job_id, branches=branches)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _preview_url(preview_path: str | None) -> str | None:
    """Map an absolute preview path under <storage>/outputs to its /files URL."""
    if not preview_path:
        return None
    outputs_root = (Path(settings.storage_local_root) / "outputs").resolve()
    try:
        rel = Path(preview_path).resolve().relative_to(outputs_root)
    except (ValueError, OSError):
        return None  # outside the served tree — never leak an arbitrary path
    return f"{FILES_URL_PREFIX}/{rel.as_posix()}"


def _resolve_preset(pipeline_id: str) -> str:
    """Return the absolute path to a YAML preset file (presets dir only).

    The path is never built from the user-supplied string: `pipeline_id` is only used
    as a key into the set of presets that actually exist, so no traversal payload can
    reach the filesystem (this is also what CodeQL's py/path-injection query wants).
    """
    presets_dir = Path("configs/presets").resolve()
    available = {p.stem: p for p in presets_dir.glob("*.yaml")}
    candidate = available.get(pipeline_id)
    if candidate is None:
        raise HTTPException(
            status_code=400,
            detail=f"Pipeline preset '{pipeline_id}' not found. "
            f"Available presets: {sorted(available)}",
        )
    return str(candidate)


def _to_response(job: JobModel) -> JobResponse:
    return JobResponse(
        id=str(job.id),
        status=job.status,
        progress=job.progress,
        pipeline_id=job.pipeline_id,
        input_path=job.input_path,
        output_path=job.output_path,
        error=job.error,
        metrics=job.metrics or {},
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
        celery_task_id=job.celery_task_id,
    )
