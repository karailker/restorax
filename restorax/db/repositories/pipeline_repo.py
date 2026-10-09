from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from restorax.core.exceptions import PipelineConfigError
from restorax.db.models import PipelineTemplateModel


class PipelineRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, pipeline: PipelineTemplateModel) -> PipelineTemplateModel:
        self._session.add(pipeline)
        await self._session.commit()
        await self._session.refresh(pipeline)
        return pipeline

    async def get(self, pipeline_id: str) -> PipelineTemplateModel:
        result = await self._session.execute(
            select(PipelineTemplateModel).where(PipelineTemplateModel.id == pipeline_id)
        )
        p = result.scalar_one_or_none()
        if p is None:
            raise PipelineConfigError(f"Pipeline '{pipeline_id}' not found")
        return p

    async def list_all(self) -> list[PipelineTemplateModel]:
        result = await self._session.execute(
            select(PipelineTemplateModel).order_by(PipelineTemplateModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def update(
        self, pipeline_id: str, name: str, description: str, config: dict[str, Any]
    ) -> PipelineTemplateModel:
        p = await self.get(pipeline_id)
        p.name = name
        p.description = description
        p.config = config
        p.updated_at = datetime.utcnow()
        await self._session.commit()
        await self._session.refresh(p)
        return p

    async def delete(self, pipeline_id: str) -> None:
        p = await self.get(pipeline_id)
        await self._session.delete(p)
        await self._session.commit()
