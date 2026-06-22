"""
RestoraX CLI — run restoration pipelines from the command line.

Usage:
  restorax run --input video.mp4 --pipeline sr_x4
  restorax run --input video.mp4 --pipeline sr_x4 --output out.mp4 --device cuda
  restorax models
  restorax presets
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from restorax.cli_download import download_models_group

console = Console()


@click.group()
def cli() -> None:
    """RestoraX — AI video restoration pipeline."""


@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Input video file")
@click.option("--pipeline", "-p", required=True, help="Pipeline preset ID (e.g. sr_x4)")
@click.option("--output", "-o", "output_path", default=None, help="Output path (default: <input>_restored.<ext>)")
@click.option("--device", default=None, help="Device override: cpu | cuda | cuda:0")
@click.option("--tile-size", default=0, show_default=True, help="Tile size for large inputs (0=no tiling)")
def run(input_path: str, pipeline: str, output_path: str | None, device: str | None, tile_size: int) -> None:
    """Run a restoration pipeline on a video file."""
    import torch

    from restorax.config import settings
    from restorax.core.pipeline import PipelineRunner, load_pipeline_from_yaml
    from restorax.core.registry import ModelRegistry
    from restorax.video.reader import VideoReader
    from restorax.video.writer import VideoWriter

    # Resolve device
    dev_str = device or settings.device
    if dev_str.startswith("cuda") and not torch.cuda.is_available():
        console.print("[yellow]CUDA unavailable, falling back to CPU[/yellow]")
        dev_str = "cpu"
    torch_device = torch.device(dev_str)

    # Resolve output path
    in_path = Path(input_path)
    if output_path is None:
        output_path = str(in_path.parent / f"{in_path.stem}_restored{in_path.suffix}")

    # Resolve preset
    preset_path = _find_preset(pipeline)
    if preset_path is None:
        console.print(f"[red]Preset '{pipeline}' not found.[/red]")
        _list_presets()
        raise SystemExit(1)

    console.print(f"[bold green]RestoraX[/bold green] | input: [cyan]{input_path}[/cyan]")
    console.print(f"  pipeline: [cyan]{preset_path}[/cyan]  device: [cyan]{torch_device}[/cyan]")

    registry = ModelRegistry(max_loaded=settings.registry_max_loaded)
    from restorax.api.routers.models import _RESTORER_CLASSES
    registry.register_all(_RESTORER_CLASSES)

    with VideoReader(input_path) as reader:
        meta = reader.meta
        console.print(f"  source:   {meta.width}×{meta.height} @ {meta.fps:.2f} fps  ({meta.frame_count} frames)")

        from restorax.core.pipeline import compute_output_fps
        pipeline_obj = load_pipeline_from_yaml(preset_path, registry)

        import math
        total_scale = math.prod(s.params.scale for s in pipeline_obj.stages if s.enabled) or 1
        out_w = meta.width * total_scale
        out_h = meta.height * total_scale
        out_fps = compute_output_fps(pipeline_obj, meta.fps)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Restoring…", total=100)

            def progress_cb(p: float) -> None:
                progress.update(task, completed=int(p * 100))

            with VideoWriter(
                output_path,
                meta=meta,
                out_width=out_w,
                out_height=out_h,
                fps=out_fps,
                source_path=input_path if meta.has_audio else None,
            ) as writer:
                PipelineRunner().run(pipeline_obj, reader, writer, progress_cb)

    fps_note = f" @ {out_fps:.2f} fps" if out_fps != meta.fps else ""
    console.print(f"\n[bold green]Done![/bold green] Output: [cyan]{output_path}[/cyan]")
    console.print(f"  resolution: {out_w}×{out_h}{fps_note}")


@cli.command(name="models")
def list_models() -> None:
    """List all available restorers."""
    from restorax.api.routers.models import _RESTORER_CLASSES

    table = Table(title="Available Restorers", show_header=True)
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("Scale")
    table.add_column("Min VRAM")
    table.add_column("Tags")

    from restorax.audio.restorer import AudioRestorerCapabilities
    for cls in _RESTORER_CLASSES:
        inst = object.__new__(cls)
        caps = cls.capabilities.fget(inst)  # type: ignore[attr-defined]
        name = cls.name.fget(inst)  # type: ignore[attr-defined]
        if isinstance(caps, AudioRestorerCapabilities):
            scale_col = "—"
            vram_col = f"{caps.min_ram_gb:.0f} GB RAM"
        else:
            scale_col = f"{caps.scale_factor}×"
            vram_col = f"{caps.min_vram_gb:.0f} GB"
        table.add_row(name, caps.category.value, scale_col, vram_col, ", ".join(caps.tags))

    console.print(table)


@cli.command(name="presets")
def list_presets_cmd() -> None:
    """List available pipeline presets."""
    _list_presets()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_preset(pipeline_id: str) -> str | None:
    candidates = [
        Path(f"configs/presets/{pipeline_id}.yaml"),
        Path(f"configs/presets/{pipeline_id}"),
        Path(pipeline_id),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _list_presets() -> None:
    preset_dir = Path("configs/presets")
    if not preset_dir.exists():
        console.print("[yellow]No presets directory found.[/yellow]")
        return
    table = Table(title="Available Presets", show_header=True)
    table.add_column("ID", style="cyan")
    table.add_column("Path")
    for p in sorted(preset_dir.glob("*.yaml")):
        table.add_row(p.stem, str(p))
    console.print(table)


@cli.group(name="benchmark")
def benchmark_group() -> None:
    """Run performance benchmarks on synthetic data."""


@benchmark_group.command(name="run")
@click.option("--restorer", "-r", default=None, help="Restorer name to benchmark (default: all)")
@click.option("--device", default="cpu", show_default=True, help="Device: cpu | cuda")
@click.option("--output-dir", "-o", default="./benchmark_results", show_default=True)
@click.option("--num-frames", default=5, show_default=True, help="Frames per degradation type")
def benchmark_run(restorer: str | None, device: str, output_dir: str, num_frames: int) -> None:
    """Benchmark restorers on synthetic degraded data. Results saved as JSON + Markdown."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from restorax.benchmarks.datasets import BenchmarkDataset
    from restorax.benchmarks.runner import BenchmarkRunner, BenchmarkSuite
    from restorax.core.registry import ModelRegistry

    from scripts.run_benchmarks import _get_all_restorer_classes, run_benchmarks
    run_benchmarks(
        restorer_name=restorer,
        device_str=device,
        output_dir=Path(output_dir),
        num_frames=num_frames,
    )


@benchmark_group.command(name="compare")
@click.argument("results_dir")
def benchmark_compare(results_dir: str) -> None:
    """Print a comparison table from previously saved benchmark results."""
    import json
    from pathlib import Path

    results_path = Path(results_dir)
    summary = results_path / "benchmark_summary.md"
    if summary.exists():
        console.print(summary.read_text())
    else:
        json_files = sorted(results_path.glob("benchmark_*.json"))
        if not json_files:
            console.print(f"[red]No benchmark files found in {results_dir}[/red]")
            return
        from restorax.benchmarks.runner import BenchmarkResult, BenchmarkSuite
        all_results = []
        for f in json_files:
            for item in json.loads(f.read_text()):
                all_results.append(BenchmarkResult(**item))
        console.print(BenchmarkSuite(results=all_results).to_markdown_table())


cli.add_command(download_models_group)


if __name__ == "__main__":
    cli()
