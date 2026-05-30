from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobModel(Base):
    __tablename__ = "jobs"

    id: Mapped[UUID] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Request fields
    input_path: Mapped[str] = mapped_column(Text, nullable=False)
    pipeline_id: Mapped[str] = mapped_column(String(128), nullable=False)
    output_format: Mapped[str] = mapped_column(String(16), nullable=False, default="mp4")
    output_codec: Mapped[str] = mapped_column(String(32), nullable=False, default="libx264")
    output_crf: Mapped[int] = mapped_column(nullable=False, default=18)
    preserve_audio: Mapped[bool] = mapped_column(nullable=False, default=True)

    # Result fields
    output_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    dag_run: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Celery
    celery_task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class PipelineTemplateModel(Base):
    __tablename__ = "pipeline_templates"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    config: Mapped[dict] = mapped_column(JSON, nullable=False)  # full YAML parsed to dict
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
