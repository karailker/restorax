from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from restorax.core.exceptions import JobNotFoundError
from restorax.db.models import JobModel


class JobRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, job: JobModel) -> JobModel:
        self._session.add(job)
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def get(self, job_id: str) -> JobModel:
        result = await self._session.execute(select(JobModel).where(JobModel.id == job_id))
        job = result.scalar_one_or_none()
        if job is None:
            raise JobNotFoundError(f"Job {job_id} not found")
        return job

    async def list_all(self, limit: int = 50, offset: int = 0) -> list[JobModel]:
        result = await self._session.execute(
            select(JobModel).order_by(JobModel.created_at.desc()).limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        job_id: str,
        status: str,
        progress: float | None = None,
        celery_task_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
        output_path: str | None = None,
        error: str | None = None,
        metrics: dict | None = None,
        dag_run: dict | None = None,
    ) -> JobModel:
        job = await self.get(job_id)
        job.status = status
        if progress is not None:
            job.progress = progress
        if celery_task_id is not None:
            job.celery_task_id = celery_task_id
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at
        if output_path is not None:
            job.output_path = output_path
        if error is not None:
            job.error = error
        if metrics is not None:
            job.metrics = metrics
        if dag_run is not None:
            job.dag_run = dag_run
        await self._session.commit()
        await self._session.refresh(job)
        return job

    async def delete(self, job_id: str) -> None:
        job = await self.get(job_id)
        await self._session.delete(job)
        await self._session.commit()
