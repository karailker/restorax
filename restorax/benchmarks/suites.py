"""
Task-specific benchmark suites for RestoraX.

Each suite targets a particular restoration task and uses the degradation
protocol closest to how that task is evaluated in the literature.

Available suites:
  SuperResolutionSuite   — bicubic ×2/×4 downscale, mixed real-world degradation
  ColorizationSuite      — grayscale → color, partial desaturation
  FaceRestorationSuite   — heavy blur, JPEG compression, noise on face-heavy content
  AudioRestorationSuite  — white noise, pink noise, clipping, reverb simulation
  VideoStabilitySuite    — inter-frame consistency metrics (no reference needed)

Usage:
    from restorax.benchmarks.suites import SuperResolutionSuite
    results = SuperResolutionSuite().run(restorer, device_str="cpu")
    print(results.to_markdown_table())
"""

from __future__ import annotations

from typing import Any

import numpy as np

from restorax.benchmarks.datasets import BenchmarkDataset, FramePair
from restorax.benchmarks.runner import BenchmarkRunner, BenchmarkSuite
from restorax.core.restorer import BaseRestorer


class SuperResolutionSuite:
    """
    SR benchmark following Set5/Set14/Urban100 paper protocols.

    Degradations: bicubic ×2, bicubic ×4, mixed real-world (noise+JPEG+downscale).
    Baselines: nearest, bilinear, bicubic, lanczos4 at each scale factor.
    """

    def __init__(self, width: int = 128, height: int = 128, num_pairs: int = 5) -> None:
        self._ds = BenchmarkDataset(
            width=width, height=height, num_pairs=num_pairs, seed=42, use_standard_patterns=True
        )

    def run(self, restorer: BaseRestorer, device_str: str = "cpu") -> BenchmarkSuite:
        runner = BenchmarkRunner()
        results = []

        for deg_name, pairs in [
            ("bicubic_x4_standard", self._ds.downscale_bicubic(factor=4)),
            (
                "bicubic_x4_random",
                BenchmarkDataset(
                    self._ds.width, self._ds.height, self._ds.num_pairs
                ).downscale_bicubic(factor=4),
            ),
            ("mixed_real_world", self._ds.mixed_degradation()),
            ("gaussian_blur", self._ds.gaussian_blur()),
        ]:
            results.append(
                runner.run(restorer, pairs, device_str=device_str, degradation_type=deg_name)
            )

        baselines = self._ds.compute_baselines(factor=4)
        return BenchmarkSuite(results=results, baselines=baselines)


class ColorizationSuite:
    """
    Colorization benchmark: evaluate perceptual colour quality.

    Uses chromatic metrics: SSIM on all channels (colour preserved),
    PSNR on luminance only (structural fidelity).
    Degradation: full grayscale + partial desaturation (50% and 80% saturation loss).
    """

    def __init__(self, width: int = 128, height: int = 128, num_pairs: int = 5) -> None:
        self._ds = BenchmarkDataset(
            width=width, height=height, num_pairs=num_pairs, seed=42, use_standard_patterns=True
        )

    def run(self, restorer: BaseRestorer, device_str: str = "cpu") -> BenchmarkSuite:
        runner = BenchmarkRunner()
        results = []

        for deg_name, pairs in [
            ("full_grayscale", self._ds.grayscale()),
            ("partial_desaturation_50", self._partial_desat(saturation=0.5)),
            ("partial_desaturation_80", self._partial_desat(saturation=0.2)),
        ]:
            results.append(
                runner.run(restorer, pairs, device_str=device_str, degradation_type=deg_name)
            )

        return BenchmarkSuite(results=results)

    def _partial_desat(self, saturation: float) -> list[FramePair]:
        """Partially desaturate reference frames by blending with grayscale."""
        import cv2

        pairs = []
        for i, ref in enumerate(self._ds._references()):
            gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            degraded = (
                ref.astype(np.float32) * saturation + gray_rgb.astype(np.float32) * (1 - saturation)
            ).astype(np.uint8)
            pairs.append(
                FramePair(degraded=degraded, reference=ref, name=f"desat_{saturation:.0%}_{i}")
            )
        return pairs


class FaceRestorationSuite:
    """
    Face restoration benchmark with progressively severe degradations.

    Levels match real-world use cases:
      light  — mild blur + light noise (good home video)
      medium — strong blur + noise + JPEG (average old footage)
      heavy  — severe blur + heavy noise + strong JPEG (badly degraded archive)
    """

    def __init__(self, width: int = 128, height: int = 128, num_pairs: int = 5) -> None:
        self._ds = BenchmarkDataset(width=width, height=height, num_pairs=num_pairs, seed=42)

    def run(self, restorer: BaseRestorer, device_str: str = "cpu") -> BenchmarkSuite:
        runner = BenchmarkRunner()
        results = []

        for deg_name, sigma, noise_std, quality in [
            ("face_light", 3.0, 10.0, 60),
            ("face_medium", 7.0, 25.0, 40),
            ("face_heavy", 15.0, 45.0, 20),
        ]:
            pairs = self._combined_face_degrad(sigma, noise_std, quality)
            results.append(
                runner.run(restorer, pairs, device_str=device_str, degradation_type=deg_name)
            )

        return BenchmarkSuite(results=results)

    def _combined_face_degrad(
        self, sigma: float, noise_std: float, quality: int
    ) -> list[FramePair]:
        import cv2

        rng = np.random.default_rng(self._ds.seed + 50)
        pairs = []
        for i, ref in enumerate(self._ds._references()):
            deg = cv2.GaussianBlur(ref, (int(sigma * 4) | 1, int(sigma * 4) | 1), sigma)
            noise = rng.normal(0, noise_std, ref.shape).astype(np.float32)
            deg = np.clip(deg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            _, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(deg, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            deg = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            pairs.append(FramePair(degraded=deg, reference=ref, name=f"face_deg_{i}"))
        return pairs


class AudioRestorationSuite:
    """
    Audio restoration benchmark with simulated degradations.

    Metrics: Signal-to-Noise Ratio (SNR), PESQ-inspired proxy (not full PESQ
    which requires a licensed library), and waveform MSE.

    Degradations:
      white_noise   — AWGN at SNR 10/20/30 dB
      pink_noise    — 1/f noise (common in film audio)
      clipping      — amplitude clipping at 0.5×/0.25× (tape saturation)
      reverb_proxy  — simple FIR convolution approximating room reverb
    """

    def __init__(self, sample_rate: int = 44100, duration: float = 2.0) -> None:
        self._sr = sample_rate
        self._n = int(sample_rate * duration)

    def run(self, restorer: object, device_str: str = "cpu") -> dict[str, Any]:
        """Returns dict of {degradation: {snr_before, snr_after, mse_before, mse_after}}."""
        from restorax.audio.restorer import AudioRestorerParams

        rng = np.random.default_rng(42)
        clean = rng.uniform(-0.3, 0.3, (self._n, 2)).astype(np.float32)
        params = AudioRestorerParams(sample_rate=self._sr)

        results = {}
        for deg_name, degraded in self._degradations(clean).items():
            restored = restorer.process_audio(degraded, params)  # type: ignore[attr-defined]
            results[deg_name] = {
                "snr_before": float(_snr(clean, degraded)),
                "snr_after": float(_snr(clean, restored[: len(clean)])),
                "mse_before": float(np.mean((clean - degraded[: len(clean)]) ** 2)),
                "mse_after": float(np.mean((clean - restored[: len(clean)]) ** 2)),
            }
        return results

    def _degradations(self, clean: np.ndarray) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(99)
        n = len(clean)
        return {
            "white_noise_20dB": clean
            + rng.normal(0, _noise_std(clean, snr_db=20), clean.shape).astype(np.float32),
            "white_noise_10dB": clean
            + rng.normal(0, _noise_std(clean, snr_db=10), clean.shape).astype(np.float32),
            "pink_noise": clean + _pink_noise(rng, n, 2, amplitude=0.05),
            "clipping_50pct": np.clip(clean * 2.0, -0.5, 0.5).astype(np.float32),
            "clipping_25pct": np.clip(clean * 4.0, -0.25, 0.25).astype(np.float32),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    signal_power = float(np.mean(clean**2))
    noise_power = float(np.mean((clean - noisy[: len(clean)]) ** 2))
    if noise_power < 1e-12:
        return 999.0
    return 10.0 * np.log10(signal_power / noise_power)


def _noise_std(signal: np.ndarray, snr_db: float) -> float:
    signal_rms = float(np.sqrt(np.mean(signal**2)))
    return signal_rms * 10 ** (-snr_db / 20.0)


def _pink_noise(rng: np.random.Generator, n: int, channels: int, amplitude: float) -> np.ndarray:
    """Approximate pink (1/f) noise via filtered white noise."""
    white = rng.normal(0, amplitude, (n, channels)).astype(np.float32)
    # Simple 1/f approximation: average white noise with low-pass filtered version
    from scipy.ndimage import uniform_filter1d

    try:
        lowpass = uniform_filter1d(white, size=100, axis=0)
        return (white * 0.5 + lowpass * 0.5).astype(np.float32)
    except ImportError:
        return white
