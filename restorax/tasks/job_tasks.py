"""
Celery tasks for RestoraX job execution.

Each task runs on a GPU worker process. The worker maintains a module-level
ModelRegistry that persists across tasks (warm LRU cache).
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import structlog

import torch
from celery import Task

from restorax.config import settings
from restorax.core.pipeline import PipelineRunner, compute_output_fps, load_pipeline_from_yaml
from restorax.core.registry import ModelRegistry
from restorax.tasks.celery_app import celery_app
from restorax.tasks.progress import ProgressReporter
from restorax.video.reader import VideoReader
from restorax.video.writer import VideoWriter

logger = structlog.get_logger(__name__)

# Module-level registry — lives for the lifetime of the worker process
_registry: ModelRegistry | None = None


def _get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
        from restorax.restorers.face_restoration.dicface import DicFaceRestorer
        from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
        from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
        from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
        from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer
        from restorax.restorers.colorization.ddcolor import DDColorRestorer
        from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
        from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer
        from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
        from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
        from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer
        from restorax.restorers.frame_interpolation.rife import RIFERestorer
        from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer
        from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
        from restorax.restorers.stabilization.gavs import GaVSRestorer
        from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer
        from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
        from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
        from restorax.restorers.super_resolution.tdm import TDMRestorer
        from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
        from restorax.restorers.super_resolution.vrt import VRTRestorer

        from restorax.core.plugin import register_plugins

        _registry = ModelRegistry(max_loaded=settings.registry_max_loaded)
        for cls in [
            RealESRGANx4Restorer, BasicVSRPlusPlusRestorer, UpscaleAVideoRestorer,
            VRTRestorer, MambaIRRestorer, TDMRestorer, SeedVRRestorer,
            Waifu2xRestorer, FlashVSRRestorer, EvTextureRestorer,
            CodeFormerRestorer, CodeFormerPlusPlusRestorer, GFPGANRestorer, DicFaceRestorer,
            DDColorRestorer, RIFERestorer,
            ScratchRemovalRestorer, HDRTVDMRestorer, VideoStabilizationRestorer,
            GaVSRestorer, AIDeinterlaceRestorer, YadifDeinterlaceRestorer,
        ]:
            _registry.register(cls)

        # Auto-discover and register third-party plugin restorers
        register_plugins(_registry)
    return _registry


# ── Module-level audio registry ───────────────────────────────────────────────

_audio_registry: object | None = None


def _get_audio_registry() -> object:
    global _audio_registry
    if _audio_registry is None:
        from restorax.audio.pipeline import AudioModelRegistry
        from restorax.restorers.audio.demucs import DemucsRestorer
        from restorax.restorers.audio.rnnoise import RNNoiseRestorer
        from restorax.restorers.audio.voicefixer import VoiceFixerRestorer

        reg = AudioModelRegistry(max_loaded=2)
        reg.register(DemucsRestorer)
        reg.register(VoiceFixerRestorer)
        reg.register(RNNoiseRestorer)
        _audio_registry = reg
    return _audio_registry


# ── Structlog context signals ─────────────────────────────────────────────────

from celery.signals import task_failure, task_postrun, task_prerun


@task_prerun.connect
def _on_task_prerun(task_id: str, task: object, args: tuple, kwargs: dict, **_: object) -> None:
    job_id = kwargs.get("job_id") or (args[0] if args else None)
    structlog.contextvars.clear_contextvars()
    ctx: dict[str, str] = {"celery_task_id": task_id}
    if job_id:
        ctx["job_id"] = str(job_id)
    structlog.contextvars.bind_contextvars(**ctx)


@task_postrun.connect
def _on_task_postrun(**_: object) -> None:
    structlog.contextvars.clear_contextvars()


@task_failure.connect
def _on_task_failure(**_: object) -> None:
    structlog.contextvars.clear_contextvars()


def _update_job_db(job_id: str, **kwargs: object) -> None:
    """Synchronous wrapper to update job status in the DB from a Celery worker."""
    import asyncio

    from restorax.db.repositories.job_repo import JobRepository
    from restorax.db.session import AsyncSessionLocal

    async def _do() -> None:
        async with AsyncSessionLocal() as session:
            repo = JobRepository(session)
            await repo.update_status(job_id, **kwargs)  # type: ignore[arg-type]

    asyncio.run(_do())


class JobTask(Task):  # type: ignore[type-arg]
    """Base task class with DB + Redis status updates on failure."""

    abstract = True

    def on_failure(self, exc: Exception, task_id: str, args: tuple, kwargs: dict, einfo: object) -> None:
        job_id = kwargs.get("job_id") or (args[0] if args else None)
        if job_id:
            try:
                _update_job_db(str(job_id), status="failed", error=str(exc))
            except Exception:
                pass
            ProgressReporter(str(job_id)).fail(str(exc))
        try:
            from restorax.telemetry import get_active_jobs_counter, get_jobs_counter
            _jc = get_jobs_counter()
            _ac = get_active_jobs_counter()
            if _jc is not None:
                _jc.add(1, {"status": "failed"})
            if _ac is not None:
                _ac.add(-1, {"pipeline": ""})
        except Exception:
            pass
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(bind=True, base=JobTask, name="restorax.tasks.job_tasks.run_job")
def run_job(
    self: Task,
    job_id: str,
    pipeline_preset_path: str,
    input_path: str,
    output_path: str,
    restore_audio: bool = False,
) -> dict:
    """
    Execute a restoration pipeline on a video file.

    Args:
        job_id: Database job UUID string.
        pipeline_preset_path: Absolute path to a YAML preset file.
        input_path: Absolute path to the input video.
        output_path: Absolute path for the output video.
        restore_audio: If True, run the audio pipeline (from audio_stages in preset)
                       and remux processed audio into the output. Default False.
    """
    reporter = ProgressReporter(job_id)
    _start_time = time.perf_counter()
    try:
        from restorax.telemetry import get_active_jobs_counter
        ctr = get_active_jobs_counter()
        if ctr is not None:
            ctr.add(1, {"pipeline": pipeline_preset_path})
    except Exception:
        pass
    _update_job_db(job_id, status="running", started_at=datetime.now(timezone.utc))
    reporter.update(0.0, status="running")

    device_str = settings.device
    if device_str.startswith("cuda") and "CUDA_VISIBLE_DEVICES" in os.environ:
        device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    registry = _get_registry()

    logger.info("job started", device=str(device), preset=pipeline_preset_path, restore_audio=restore_audio)

    with VideoReader(input_path) as reader:
        meta = reader.meta
        scale = _infer_output_scale(pipeline_preset_path)
        out_w = meta.width * scale
        out_h = meta.height * scale

        pipeline = load_pipeline_from_yaml(pipeline_preset_path, registry)
        runner = PipelineRunner()

        # Compute output fps — RIFE doubles it; most restorers leave it unchanged
        out_fps = compute_output_fps(pipeline, meta.fps)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # When audio restoration is requested, skip passthrough so processed audio
        # can be added in the second pass (prevents double audio streams).
        passthrough_audio = meta.has_audio and not restore_audio

        with VideoWriter(
            output_path,
            meta=meta,
            out_width=out_w,
            out_height=out_h,
            fps=out_fps,
            source_path=input_path if passthrough_audio else None,
        ) as writer:
            runner.run(
                pipeline,
                reader,
                writer,
                progress_cb=lambda p: reporter.update(p, status="running"),
            )

    # Optional audio restoration pass
    if restore_audio and meta.has_audio:
        try:
            _run_audio_pipeline(pipeline_preset_path, input_path, output_path, device)
        except Exception as exc:
            logger.warning("Audio pipeline failed (%s) — video output kept as-is", exc)

    _update_job_db(
        job_id, status="completed",
        progress=1.0, output_path=output_path,
        completed_at=datetime.now(timezone.utc),
    )
    reporter.complete(output_path)
    logger.info("job completed", output_path=output_path)
    try:
        from restorax.telemetry import (
            get_active_jobs_counter,
            get_job_duration_histogram,
            get_jobs_counter,
        )
        _dur = time.perf_counter() - _start_time
        _pipeline_name = Path(pipeline_preset_path).stem
        _jc = get_jobs_counter()
        _jd = get_job_duration_histogram()
        _ac = get_active_jobs_counter()
        if _jc is not None:
            _jc.add(1, {"status": "completed"})
        if _jd is not None:
            _jd.record(_dur, {"pipeline": _pipeline_name})
        if _ac is not None:
            _ac.add(-1, {"pipeline": pipeline_preset_path})
    except Exception:
        pass
    return {"output_path": output_path, "metrics": {}}


@celery_app.task(bind=True, base=JobTask, name="restorax.tasks.job_tasks.run_dag_job")
def run_dag_job(
    self,
    job_id: str,
    dag_id: str,
    input_path: str,
    output_path: str,
) -> dict:
    """
    Execute a DAG pipeline on a video file.
    Loads the DAG from DB, builds ExecutionContext, runs DAGExecutor.
    No Celery canvas — all orchestration is internal.
    """
    import asyncio as _asyncio
    import uuid as _uuid

    from restorax.dag import DAGExecutor
    from restorax.dag.context import ExecutionContext, ProgressEmitter
    from restorax.dag.serializer import DAGSerializer
    from restorax.dag.nodes import control, io, map_node, merge, parallel, restore  # noqa: F401 — registers node types

    reporter = ProgressReporter(job_id)
    _update_job_db(job_id, status="running", started_at=datetime.now(timezone.utc))
    reporter.update(0.0, status="running")

    device_str = settings.device
    if device_str.startswith("cuda") and "CUDA_VISIBLE_DEVICES" in os.environ:
        device_str = "cuda:0"
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")

    async def _load_dag():
        from restorax.db.repositories.pipeline_repo import PipelineRepository
        from restorax.db.session import AsyncSessionLocal
        from restorax.core.exceptions import PipelineConfigError
        async with AsyncSessionLocal() as session:
            repo = PipelineRepository(session)
            try:
                template = await repo.get(dag_id)
            except PipelineConfigError:
                raise ValueError(f"DAG '{dag_id}' not found in database")
        return DAGSerializer.from_dict(template.config)

    dag = _asyncio.run(_load_dag())

    work_dir = Path(output_path).parent / "dag_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    emitter = ProgressEmitter(job_id=job_id, redis_url=settings.redis_url)
    ctx = ExecutionContext(
        run_id=str(_uuid.uuid4()),
        job_id=job_id,
        work_dir=work_dir,
        device=device,
        registry=_get_registry(),
        progress_emitter=emitter,
        logger=logger,
        config={"input_path": input_path, "output_path": output_path},
    )

    dag_run = _asyncio.run(DAGExecutor().execute(dag, ctx))
    _update_job_db(job_id, status="running", dag_run=dag_run.to_dict())

    if not dag_run.succeeded:
        raise RuntimeError(dag_run.error or "DAG execution failed")

    _update_job_db(
        job_id, status="completed",
        progress=1.0, output_path=output_path,
        completed_at=datetime.now(timezone.utc),
    )
    reporter.complete(output_path)
    return {"output_path": output_path}


def _run_audio_pipeline(
    preset_path: str,
    input_path: str,
    output_path: str,
    device: torch.device,
) -> None:
    """Parse audio_stages from preset and run audio restoration, remuxing into output."""
    import yaml

    with open(preset_path) as f:
        config = yaml.safe_load(f)

    audio_stages_cfg = config.get("audio_stages", [])
    if not audio_stages_cfg:
        logger.debug("no audio_stages in preset, skipping")
        return

    from restorax.audio.pipeline import (
        AudioModelRegistry, AudioPipeline, AudioPipelineRunner, AudioStage,
        load_audio_pipeline_from_config,
    )
    from restorax.audio.reader import AudioReader
    from restorax.audio.restorer import AudioRestorerParams
    from restorax.audio.writer import AudioWriter

    audio_arr, sr = AudioReader(input_path).read()
    audio_registry = _get_audio_registry()
    assert isinstance(audio_registry, AudioModelRegistry)

    pipeline = load_audio_pipeline_from_config(config, audio_registry, device)
    if pipeline is None:
        return

    processed = AudioPipelineRunner().run(pipeline, audio_arr, sr)
    AudioWriter().mux_into_video(output_path, processed, sr, output_path)
    logger.info("audio pipeline complete", stages=len(pipeline.stages))


def _infer_output_scale(preset_path: str) -> int:
    """Quick parse of the preset YAML to get overall output scale."""
    import yaml

    try:
        with open(preset_path) as f:
            config = yaml.safe_load(f)
        scale = 1
        for stage in config.get("stages", []):
            scale *= stage.get("scale", 1)
        return max(scale, 1)
    except Exception:
        return 1
