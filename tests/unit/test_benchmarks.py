"""Tests for the benchmark system: dataset, runner, vram_monitor, suite output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from restorax.benchmarks.datasets import BenchmarkDataset
from restorax.benchmarks.runner import BenchmarkResult, BenchmarkRunner, BenchmarkSuite
from restorax.benchmarks.vram_monitor import VRAMMonitor
from tests.conftest import IdentityRestorer

# ── BenchmarkDataset ──────────────────────────────────────────────────────────


class TestBenchmarkDataset:
    @pytest.fixture
    def ds(self):
        return BenchmarkDataset(width=64, height=64, num_pairs=3, seed=42)

    def test_all_degradations_returns_eight_types_base(self, ds):
        all_degs = ds.all_degradations()
        assert len(all_degs) == 8
        expected = {
            "gaussian_blur",
            "add_noise",
            "jpeg_compress",
            "downscale_bicubic",
            "simulate_scratch",
            "grayscale",
            "mixed_degradation",
            "standard_patterns",
        }
        assert set(all_degs.keys()) == expected

    def test_pair_count_matches_num_pairs(self, ds):
        for name, pairs in ds.all_degradations().items():
            assert len(pairs) == 3, f"{name}: expected 3 pairs, got {len(pairs)}"

    def test_degraded_shape(self, ds):
        for pairs in ds.all_degradations().values():
            for pair in pairs:
                assert pair.degraded.shape == (64, 64, 3)
                assert pair.degraded.dtype == np.uint8

    def test_reference_shape(self, ds):
        for pairs in ds.all_degradations().values():
            for pair in pairs:
                assert pair.reference.shape == (64, 64, 3)
                assert pair.reference.dtype == np.uint8

    def test_degraded_differs_from_reference(self, ds):
        """Degraded frame must differ from its reference (degradation was applied)."""
        blur_pairs = ds.gaussian_blur()
        for pair in blur_pairs:
            assert not np.array_equal(pair.degraded, pair.reference)

    def test_reproducibility(self):
        ds1 = BenchmarkDataset(width=32, height=32, num_pairs=2, seed=42)
        ds2 = BenchmarkDataset(width=32, height=32, num_pairs=2, seed=42)
        p1 = ds1.gaussian_blur()
        p2 = ds2.gaussian_blur()
        assert np.array_equal(p1[0].reference, p2[0].reference)

    def test_downscale_output_size_unchanged(self, ds):
        """Downscaled frames are resized back to original dimensions."""
        pairs = ds.downscale_bicubic(factor=4)
        assert pairs[0].degraded.shape == (64, 64, 3)

    def test_all_degradations_returns_eight_types(self, ds):
        """New: mixed_degradation and standard_patterns are included."""
        all_degs = ds.all_degradations()
        assert len(all_degs) == 8
        assert "mixed_degradation" in all_degs
        assert "standard_patterns" in all_degs

    def test_downscale_lr_pairs_scale_factor(self, ds):
        """LR pairs have correct scale_factor metadata."""
        pairs = ds.downscale_lr_pairs(factor=4)
        assert pairs[0].scale_factor == 4
        assert pairs[0].degraded.shape == (16, 16, 3)  # 64 / 4
        assert pairs[0].reference.shape == (64, 64, 3)

    def test_standard_patterns_mode(self):
        ds = BenchmarkDataset(width=64, height=64, num_pairs=3, seed=42, use_standard_patterns=True)
        pairs = ds.standard_patterns(factor=4)
        assert len(pairs) == 3
        assert pairs[0].dataset == "standard_patterns"

    def test_compute_baselines_returns_all_methods(self, ds):
        baselines = ds.compute_baselines(factor=4)
        for method in ("nearest", "bilinear", "bicubic", "lanczos4", "sharpened_bicubic"):
            assert method in baselines
            assert baselines[method].psnr > 0
            assert 0.0 < baselines[method].ssim <= 1.0

    def test_bicubic_beats_nearest(self, ds):
        """Bicubic should always beat nearest-neighbour on smooth content."""
        baselines = ds.compute_baselines(factor=4)
        assert baselines["bicubic"].psnr >= baselines["nearest"].psnr


# ── BenchmarkRunner ───────────────────────────────────────────────────────────


class TestBenchmarkRunner:
    @pytest.fixture
    def restorer(self):
        r = IdentityRestorer(scale=1)
        r.load(torch.device("cpu"))
        return r

    @pytest.fixture
    def pairs(self):
        return BenchmarkDataset(width=32, height=32, num_pairs=2).gaussian_blur()

    def test_run_returns_benchmark_result(self, restorer, pairs):
        result = BenchmarkRunner().run(
            restorer, pairs, device_str="cpu", degradation_type="gaussian_blur"
        )
        assert isinstance(result, BenchmarkResult)

    def test_result_fields_finite(self, restorer, pairs):
        result = BenchmarkRunner().run(
            restorer, pairs, device_str="cpu", degradation_type="gaussian_blur"
        )
        assert result.num_frames == 2
        assert result.fps > 0
        assert result.vram_peak_mb == 0.0  # CPU device — no CUDA allocations
        assert result.device == "cpu"

    def test_psnr_ssim_in_valid_range(self, restorer, pairs):
        result = BenchmarkRunner().run(restorer, pairs, device_str="cpu")
        assert result.psnr_mean >= 0
        assert 0.0 <= result.ssim_mean <= 1.0

    def test_run_all_degradations_returns_suite(self, restorer):
        ds = BenchmarkDataset(width=32, height=32, num_pairs=1)
        suite = BenchmarkRunner().run_all_degradations(restorer, ds, device_str="cpu")
        assert isinstance(suite, BenchmarkSuite)
        assert len(suite.results) == 8  # one per degradation type (now 8)


# ── BenchmarkSuite output ─────────────────────────────────────────────────────


class TestBenchmarkSuite:
    @pytest.fixture
    def suite(self):
        results = [
            BenchmarkResult(
                "real_esrgan_x4plus",
                "gaussian_blur",
                28.4,
                0.821,
                0.123,
                12.3,
                4096.0,
                "cuda",
                5,
                "2026-04-25T00:00:00+00:00",
            ),
            BenchmarkResult(
                "mamba_ir_x4",
                "add_noise",
                29.1,
                0.835,
                0.118,
                18.7,
                3072.0,
                "cuda",
                5,
                "2026-04-25T00:00:00+00:00",
            ),
        ]
        return BenchmarkSuite(results=results)

    def test_to_json_is_valid(self, suite):
        import json

        data = json.loads(suite.to_json())
        # New structure: {"results": [...], "baselines": {...}}
        assert "results" in data
        assert len(data["results"]) == 2
        assert data["results"][0]["restorer_name"] == "real_esrgan_x4plus"

    def test_markdown_table_has_header(self, suite):
        table = suite.to_markdown_table()
        assert "| Restorer |" in table
        assert "PSNR" in table

    def test_markdown_row_format(self, suite):
        row = suite.results[0].to_markdown_row()
        assert "real_esrgan_x4plus" in row
        assert "gaussian_blur" in row
        assert "28.4" in row

    def test_save_creates_files(self, suite, tmp_path: Path):
        suite.save(tmp_path, name="test_bench")
        assert (tmp_path / "test_bench.json").exists()
        assert (tmp_path / "test_bench.md").exists()


# ── VRAMMonitor ───────────────────────────────────────────────────────────────


class TestVRAMMonitor:
    def test_does_not_crash_on_cpu(self):
        with VRAMMonitor() as mon:
            _ = np.zeros((1000, 1000))
        assert mon.peak_mb < 1.0  # no new CUDA allocs for pure numpy work

    def test_context_manager_protocol(self):
        mon = VRAMMonitor()
        with mon:
            pass
        assert isinstance(mon.peak_mb, float)

    def test_nested_monitors(self):
        with VRAMMonitor() as m1, VRAMMonitor() as m2:
            pass
        assert m2.peak_mb >= 0.0
        assert m1.peak_mb >= 0.0
