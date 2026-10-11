#!/usr/bin/env python3
"""
Run RestoraX benchmarks against all (or selected) restorers.

Measures PSNR, SSIM, LPIPS, speed (fps), and peak VRAM for each
restorer × degradation type on synthetic test data designed to
replicate the evaluation protocols used in SR literature (Set5/Set14
bicubic ×4, Real-ESRGAN blind degradation, etc.).

Also computes classical baselines (nearest, bilinear, bicubic, lanczos4)
so results can be directly contextualised against paper baselines.

Output:
  {output_dir}/benchmark_{restorer_name}.json   per-restorer detailed results
  {output_dir}/benchmark_summary.md             Markdown table with baselines
  {output_dir}/benchmark_summary.json           combined JSON

Usage:
    # All restorers, CPU (fast smoke test)
    python scripts/run_benchmarks.py

    # Standard-pattern test images (Lena/Cameraman-style), CUDA
    python scripts/run_benchmarks.py --device cuda --standard-patterns

    # Single restorer, CUDA
    python scripts/run_benchmarks.py --restorer real_esrgan_x4plus --device cuda

    # Custom output directory
    python scripts/run_benchmarks.py --output-dir /tmp/bench --num-frames 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from restorax.benchmarks.datasets import BenchmarkDataset
from restorax.benchmarks.runner import BenchmarkResult, BenchmarkRunner, BenchmarkSuite
from restorax.core.registry import ModelRegistry


def _get_all_restorer_classes() -> list:
    from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
    from restorax.restorers.colorization.ddcolor import DDColorRestorer
    from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
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

    return [
        RealESRGANx4Restorer,
        MambaIRRestorer,
        BasicVSRPlusPlusRestorer,
        VRTRestorer,
        UpscaleAVideoRestorer,
        TDMRestorer,
        CodeFormerRestorer,
        CodeFormerPlusPlusRestorer,
        GFPGANRestorer,
        DDColorRestorer,
        RIFERestorer,
        ScratchRemovalRestorer,
        HDRTVDMRestorer,
        VideoStabilizationRestorer,
        GaVSRestorer,
        AIDeinterlaceRestorer,
    ]


def run_benchmarks(
    restorer_name: str | None,
    device_str: str,
    output_dir: Path,
    num_frames: int,
    standard_patterns: bool = False,
) -> None:
    device = torch.device(device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu")
    dataset = BenchmarkDataset(
        width=128,
        height=128,
        num_pairs=num_frames,
        seed=42,
        use_standard_patterns=standard_patterns,
    )
    runner = BenchmarkRunner(half_precision=False)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute classical baselines once (no GPU needed)
    print("\nComputing classical baselines (nearest / bilinear / bicubic / lanczos4)…")
    baselines = dataset.compute_baselines(factor=4)
    for name, bl in baselines.items():
        print(f"  {name:<22} PSNR={bl.psnr:.2f} dB  SSIM={bl.ssim:.4f}  {bl.fps:.0f}+ fps")

    all_classes = _get_all_restorer_classes()
    if restorer_name:
        all_classes = [c for c in all_classes if c().name == restorer_name]  # type: ignore[misc]
        if not all_classes:
            print(f"\nRestorer '{restorer_name}' not found. Available:")
            print("  " + "\n  ".join(c().name for c in _get_all_restorer_classes()))  # type: ignore[misc]
            sys.exit(1)

    all_results: list[BenchmarkResult] = []
    for cls in all_classes:
        registry = ModelRegistry(max_loaded=1)
        registry.register(cls)
        try:
            restorer = registry.get(cls().name, device)  # type: ignore[misc]
        except Exception as exc:
            print(f"\nSkipping {cls.__name__}: {exc}")
            continue

        print(f"\nBenchmarking: {restorer.name} on {device_str}")
        try:
            suite = runner.run_all_degradations(
                restorer, dataset, device_str=device_str, include_baselines=False
            )
        except Exception as exc:
            print(f"  ERROR: {exc}")
            registry.unload_all()
            continue
        all_results.extend(suite.results)

        print(suite.to_markdown_table(include_baselines=False))

        per_file = output_dir / f"benchmark_{restorer.name}.json"
        per_file.write_text(suite.to_json())
        print(f"  Saved: {per_file}")

        registry.unload_all()

    # Save combined summary with baselines
    summary_suite = BenchmarkSuite(results=all_results, baselines=baselines)
    summary_suite.save(output_dir, name="benchmark_summary")
    print(f"\nSummary (with baselines) saved to {output_dir}/benchmark_summary.{{json,md}}")
    print("\n" + summary_suite.to_markdown_table())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RestoraX benchmarks")
    parser.add_argument(
        "--restorer", "-r", default=None, help="Single restorer name (default: all)"
    )
    parser.add_argument("--device", default="cpu", help="Device: cpu | cuda")
    parser.add_argument("--output-dir", "-o", default="./benchmark_results")
    parser.add_argument("--num-frames", type=int, default=5, help="Frames per degradation type")
    parser.add_argument(
        "--standard-patterns",
        action="store_true",
        help="Use Lena/Cameraman/Baboon/Urban-style test images instead of random frames",
    )
    args = parser.parse_args()

    run_benchmarks(
        restorer_name=args.restorer,
        device_str=args.device,
        output_dir=Path(args.output_dir),
        num_frames=args.num_frames,
        standard_patterns=args.standard_patterns,
    )


if __name__ == "__main__":
    main()
