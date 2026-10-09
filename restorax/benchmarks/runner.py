"""
BenchmarkRunner — measures restorer quality and performance.

Metrics collected per restorer × degradation_type:
  psnr_mean    PSNR (dB) vs clean reference, higher is better
  ssim_mean    SSIM [0,1] vs clean reference, higher is better
  lpips_mean   LPIPS [0,∞) vs clean reference, lower is better
  fps          frames processed per second (wall time)
  vram_peak_mb peak GPU memory allocated (0.0 on CPU)

Also computes classical baseline comparisons (nearest, bilinear, bicubic,
lanczos4, sharpened_bicubic) so RestoraX results can be contextualised
against standard SR paper baselines from SRCNN / EDSR / Real-ESRGAN.

Output: BenchmarkResult / BenchmarkSuite → JSON + Markdown table.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from restorax.benchmarks.datasets import BaselineResult, BenchmarkDataset, FramePair
from restorax.benchmarks.vram_monitor import VRAMMonitor
from restorax.core.restorer import BaseRestorer, RestorerParams


@dataclass
class BenchmarkResult:
    restorer_name: str
    degradation_type: str
    psnr_mean: float
    ssim_mean: float
    lpips_mean: float
    fps: float
    vram_peak_mb: float
    device: str
    num_frames: int
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown_row(self) -> str:
        lpips = f"{self.lpips_mean:.3f}" if self.lpips_mean > 0 else "—"
        psnr = f"{self.psnr_mean:.1f}" if self.psnr_mean < 999 else "∞"
        vram = f"{self.vram_peak_mb / 1024:.1f} GB" if self.vram_peak_mb > 0 else "CPU"
        return (
            f"| `{self.restorer_name}` | {self.degradation_type} "
            f"| {psnr} | {self.ssim_mean:.3f} | {lpips} "
            f"| {self.fps:.1f} | {vram} |"
        )


@dataclass
class BenchmarkSuite:
    results: list[BenchmarkResult]
    baselines: dict[str, BaselineResult] = None  # type: ignore[assignment]

    def to_json(self) -> str:
        data: dict[str, Any] = {"results": [r.to_dict() for r in self.results]}
        if self.baselines:
            from dataclasses import asdict

            data["baselines"] = {k: asdict(v) for k, v in self.baselines.items()}
        return json.dumps(data, indent=2)

    def to_markdown_table(self, include_baselines: bool = True) -> str:
        header = (
            "| Restorer | Degradation | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Speed (fps) | VRAM |\n"
            "|---|---|---|---|---|---|---|"
        )
        rows = []

        # Classical baselines first (for context)
        if include_baselines and self.baselines:
            rows.append("| **Classical baselines** | | | | | | |")
            for name, bl in self.baselines.items():
                rows.append(
                    f"| {name} | bicubic ×4 | {bl.psnr:.1f} | {bl.ssim:.3f} | — "
                    f"| {bl.fps:.0f}+ | CPU |"
                )
            rows.append("| **RestoraX restorers** | | | | | | |")

        rows.extend(r.to_markdown_row() for r in self.results)
        return "\n".join([header] + rows)

    def save(self, output_dir: str | Path, name: str = "benchmark") -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(self.to_json())
        (out / f"{name}.md").write_text(self.to_markdown_table())


class BenchmarkRunner:
    """
    Run a BaseRestorer against a list of FramePairs and collect metrics.

    Usage:
        runner = BenchmarkRunner()
        pairs = BenchmarkDataset(num_pairs=5).gaussian_blur()
        result = runner.run(restorer, pairs, device_str="cpu")
    """

    def __init__(self, half_precision: bool = False) -> None:
        self._half = half_precision

    def run(
        self,
        restorer: BaseRestorer,
        pairs: list[FramePair],
        device_str: str = "cpu",
        degradation_type: str = "unknown",
    ) -> BenchmarkResult:
        from restorax.metrics.full_reference import lpips as compute_lpips
        from restorax.metrics.full_reference import psnr as compute_psnr
        from restorax.metrics.full_reference import ssim as compute_ssim

        device = torch.device(device_str)
        params = RestorerParams(half_precision=self._half)

        psnr_vals, ssim_vals, lpips_vals = [], [], []
        frame_times: list[float] = []

        with VRAMMonitor() as mon:
            for pair in pairs:
                t0 = time.perf_counter()

                # For scale-changing restorers (SR), the reference must be the
                # degraded input upscaled to match the output size for fair comparison
                output = restorer.process_frame(pair.degraded, params)

                frame_times.append(time.perf_counter() - t0)

                # Resize reference to output size for metric computation
                ref = pair.reference
                if output.shape[:2] != ref.shape[:2]:
                    ref = _resize_ref(ref, output.shape[1], output.shape[0])

                psnr_vals.append(compute_psnr(output, ref))
                ssim_vals.append(compute_ssim(output, ref, device=device))
                lpips_vals.append(compute_lpips(output, ref, device=device))

        total_time = sum(frame_times)
        fps = len(pairs) / total_time if total_time > 0 else 0.0

        return BenchmarkResult(
            restorer_name=restorer.name,
            degradation_type=degradation_type,
            psnr_mean=float(np.mean([v for v in psnr_vals if v < 999])) if psnr_vals else 0.0,
            ssim_mean=float(np.mean(ssim_vals)),
            lpips_mean=float(np.mean(lpips_vals)),
            fps=fps,
            vram_peak_mb=mon.peak_mb,
            device=device_str,
            num_frames=len(pairs),
            timestamp=datetime.now(tz=UTC).isoformat(),
        )

    def run_all_degradations(
        self,
        restorer: BaseRestorer,
        dataset: BenchmarkDataset,
        device_str: str = "cpu",
        include_baselines: bool = True,
    ) -> BenchmarkSuite:
        """
        Run restorer against all degradation types and compute classical baselines.

        Baselines (nearest/bilinear/bicubic/lanczos4) are computed once and included
        in the BenchmarkSuite so README tables can show how RestoraX compares to the
        standard SR paper baselines used in SRCNN/EDSR/RCAN publications.
        """
        results = []
        for deg_name, pairs in dataset.all_degradations().items():
            result = self.run(restorer, pairs, device_str=device_str, degradation_type=deg_name)
            results.append(result)

        baselines = dataset.compute_baselines() if include_baselines else None
        return BenchmarkSuite(results=results, baselines=baselines)


def _resize_ref(ref: np.ndarray, w: int, h: int) -> np.ndarray:
    import cv2

    return cv2.resize(ref, (w, h), interpolation=cv2.INTER_CUBIC)
