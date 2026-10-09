"""
BenchmarkDataset — reproducible test pairs for video/image restoration benchmarking.

Standard benchmark suites used in the SR/restoration literature:

  SYNTHETIC (generated programmatically — no copyright, fully reproducible):
    Deterministic color-gradient frames with controlled degradations.
    Used when no external dataset is available (CI, quick smoke tests).

  STANDARD_PATTERNS (classic image processing test patterns):
    Lena/Lenna, Cameraman, Baboon, Peppers — these are public-domain images
    widely used in signal processing research since the 1970s. Included as
    synthetic reproductions with comparable statistical properties.

  LITERATURE_DEGRADATIONS (standard SR benchmark protocol from SRCNN / EDSR):
    Bicubic downscale by ×2, ×3, ×4 as used in Set5/Set14/Urban100 papers.
    Gaussian blur σ=1.0 as used in blind SR evaluation.
    JPEG quality 10–30 as used in artifact removal benchmarks.

Baseline comparison methods (classical, no GPU required):
  - nearest      : Nearest-neighbour interpolation
  - bilinear     : Bilinear interpolation
  - bicubic      : Bicubic interpolation (standard SR paper baseline)
  - lanczos      : Lanczos (Sinc-based) interpolation
  - edsr_style   : Simulated EDSR-class result via sharpened bicubic
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FramePair:
    degraded: np.ndarray  # HxWx3 uint8 — restorer input
    reference: np.ndarray  # HxWx3 uint8 — clean ground truth
    name: str  # "gaussian_blur_0", "bicubic_x4_lena", etc.
    scale_factor: int = 1  # spatial scale expected from restorer (1 or 4)
    dataset: str = "synthetic"  # "synthetic" | "standard_patterns" | "custom"


@dataclass
class BaselineResult:
    """Quality metrics for a classical upscaling baseline."""

    method: str
    psnr: float
    ssim: float
    fps: float


# ── Standard test image generators ────────────────────────────────────────────


def _lena_like(size: int = 128) -> np.ndarray:
    """
    Synthetic frame with similar statistical properties to the classic Lena image:
    smooth flesh-tone gradients, hat brim edges, and high-frequency feather texture.
    Used as a public-domain substitute for the copyrighted Lena photograph.
    """
    rng = np.random.default_rng(1973)  # year Lena was first used as test image
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    # Flesh-tone gradient background
    for y in range(size):
        for c, base in enumerate([200, 155, 120]):
            frame[y, :, c] = np.clip(base - y * 0.5, 80, 240).astype(np.uint8)
    # Hat brim: dark curved band across the upper third
    for x in range(size):
        y = int(size * 0.28 + 8 * np.sin(x / size * np.pi * 2))
        frame[max(0, y - 4) : min(size, y + 4), x] = [40, 30, 20]
    # Feathers: high-frequency texture (tests high-frequency SR recovery)
    for _ in range(120):
        x, y = int(rng.integers(size * 0.3, size)), int(rng.integers(0, size * 0.6))
        cv2.line(
            frame,
            (x, y),
            (x + int(rng.integers(-8, 8)), y + int(rng.integers(3, 10))),
            tuple(int(v) for v in rng.integers(80, 200, 3).tolist()),
            1,
        )
    return frame


def _cameraman_like(size: int = 128) -> np.ndarray:
    """Grayscale-converted scene with sharp silhouette edges (classic Cameraman substitute)."""
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    # Sky gradient
    for y in range(size // 2):
        v = int(190 + y * 0.6)
        frame[y, :] = [v, v, v + 10]
    # Ground
    frame[size // 2 :] = [80, 90, 70]
    # Tripod legs
    cx, base_y = size // 2, size - 1
    for angle_offset in [-20, 0, 20]:
        top_x = cx + int(np.tan(np.radians(angle_offset)) * (size // 3))
        cv2.line(frame, (cx, base_y - size // 3), (top_x, base_y), [30, 30, 30], 2)
    # Camera body silhouette
    cv2.rectangle(
        frame, (cx - 15, base_y - size // 3 - 20), (cx + 15, base_y - size // 3), [20, 20, 20], -1
    )
    return frame


def _baboon_like(size: int = 128) -> np.ndarray:
    """
    Colourful frame with many distinct hue regions — equivalent statistical
    challenge to the Baboon/Mandrill test image used since the 1970s.
    """
    rng = np.random.default_rng(1977)
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    # Divide into colourful blocks (simulates baboon's multi-coloured face)
    block_sz = size // 8
    colours = [
        [180, 40, 40],
        [40, 160, 40],
        [40, 40, 180],
        [160, 160, 40],
        [160, 40, 160],
        [40, 160, 160],
        [200, 100, 40],
        [100, 200, 40],
    ]
    for i in range(8):
        for j in range(8):
            c = colours[(i * 3 + j * 2) % len(colours)]
            noise = rng.integers(-20, 20, 3)
            colour = np.clip(np.array(c) + noise, 0, 255).tolist()
            frame[i * block_sz : (i + 1) * block_sz, j * block_sz : (j + 1) * block_sz] = colour
    # Add edge lines (high-frequency content)
    for i in range(1, 8):
        cv2.line(frame, (0, i * block_sz), (size, i * block_sz), [0, 0, 0], 1)
        cv2.line(frame, (i * block_sz, 0), (i * block_sz, size), [0, 0, 0], 1)
    return frame


def _urban_like(size: int = 128) -> np.ndarray:
    """
    Geometric pattern with regular line structures — inspired by Urban100
    which tests SR recovery of architectural repetition and straight edges.
    """
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    # Building facade: regular windows
    frame[:] = [150, 140, 130]  # concrete background
    for row in range(4):
        for col in range(6):
            y1 = 10 + row * 28
            x1 = 5 + col * 20
            cv2.rectangle(frame, (x1, y1), (x1 + 14, y1 + 18), [60, 80, 120], -1)  # window
            cv2.rectangle(frame, (x1 + 6, y1), (x1 + 8, y1 + 18), [180, 170, 150], 1)  # frame
    # Horizontal floor lines
    for row in range(4):
        y = 8 + row * 28
        cv2.line(frame, (0, y), (size, y), [100, 95, 90], 2)
    return frame


# ── Main dataset class ────────────────────────────────────────────────────────


class BenchmarkDataset:
    """
    Generate (degraded, reference) FramePairs for benchmarking.

    Two modes:
      - synthetic: deterministic RNG frames (default, CI-safe, fast)
      - standard_patterns: Lena-like, Cameraman-like, Baboon-like, Urban-like
                           classic test images reproduced as public-domain equivalents

    Scale factors follow the standard SR paper protocol:
      ×2, ×3, ×4 downscale via bicubic (matching EDSR / RCAN / ESRGAN papers).
    """

    def __init__(
        self,
        width: int = 128,
        height: int = 128,
        num_pairs: int = 5,
        seed: int = 42,
        use_standard_patterns: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.num_pairs = num_pairs
        self.seed = seed
        self.use_standard_patterns = use_standard_patterns

    # ── Public API ─────────────────────────────────────────────────────────────

    def gaussian_blur(self, ksize: int = 15, sigma: float = 3.0) -> list[FramePair]:
        """Gaussian blur — models camera defocus and slight motion blur."""
        pairs = []
        for i, ref in enumerate(self._references()):
            degraded = cv2.GaussianBlur(ref, (ksize | 1, ksize | 1), sigma)
            pairs.append(
                FramePair(degraded=degraded, reference=ref, name=f"gaussian_blur_s{sigma:.0f}_{i}")
            )
        return pairs

    def add_noise(self, std: float = 30.0) -> list[FramePair]:
        """Additive Gaussian noise — models sensor noise, VHS grain, film grain."""
        rng = np.random.default_rng(self.seed + 1)
        pairs = []
        for i, ref in enumerate(self._references()):
            noise = rng.normal(0, std, ref.shape).astype(np.float32)
            degraded = np.clip(ref.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            pairs.append(FramePair(degraded=degraded, reference=ref, name=f"awgn_std{std:.0f}_{i}"))
        return pairs

    def jpeg_compress(self, quality: int = 20) -> list[FramePair]:
        """JPEG compression — matches artifact removal paper protocol (quality 10–30)."""
        pairs = []
        for i, ref in enumerate(self._references()):
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, buf = cv2.imencode(".jpg", cv2.cvtColor(ref, cv2.COLOR_RGB2BGR), encode_param)
            degraded_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            degraded = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)
            pairs.append(FramePair(degraded=degraded, reference=ref, name=f"jpeg_q{quality}_{i}"))
        return pairs

    def downscale_bicubic(self, factor: int = 4) -> list[FramePair]:
        """
        Bicubic downscale by factor× — standard SR benchmark degradation.

        Matches the protocol used in Set5/Set14/BSDS100/Urban100 evaluations:
        1. Downscale HR image by ×factor with bicubic interpolation → LR
        2. Upscale LR back to HR size with bicubic → degraded (the baseline)
        3. Original HR is the reference

        scale_factor is set on the pair so BenchmarkRunner knows the restorer
        is expected to output factor× larger frames.
        """
        pairs = []
        for i, ref in enumerate(self._references()):
            lr = cv2.resize(
                ref, (self.width // factor, self.height // factor), interpolation=cv2.INTER_CUBIC
            )
            # degraded = bicubic upscale of LR (the "naive baseline" input)
            degraded = cv2.resize(lr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            pairs.append(
                FramePair(
                    degraded=degraded, reference=ref, name=f"bicubic_x{factor}_{i}", scale_factor=1
                )
            )
        return pairs

    def downscale_lr_pairs(self, factor: int = 4) -> list[FramePair]:
        """
        True LR→HR pairs: degraded is the small LR image, reference is the HR original.

        Use this when testing models that actually upscale (Real-ESRGAN, MambaIR etc.)
        rather than models that enhance an already-upscaled image.
        BenchmarkRunner resizes the reference to match restorer output for fair comparison.
        """
        pairs = []
        for i, ref in enumerate(self._references()):
            lr = cv2.resize(
                ref, (self.width // factor, self.height // factor), interpolation=cv2.INTER_CUBIC
            )
            pairs.append(
                FramePair(degraded=lr, reference=ref, name=f"lr_x{factor}_{i}", scale_factor=factor)
            )
        return pairs

    def simulate_scratch(self, num_scratches: int = 3) -> list[FramePair]:
        """Bright vertical streaks — film scratch / tape dropout simulation."""
        rng = np.random.default_rng(self.seed + 2)
        pairs = []
        for i, ref in enumerate(self._references()):
            degraded = ref.copy()
            for _ in range(num_scratches):
                x = int(rng.integers(5, self.width - 5))
                degraded[:, x : x + 2] = 240
            pairs.append(FramePair(degraded=degraded, reference=ref, name=f"scratch_{i}"))
        return pairs

    def grayscale(self) -> list[FramePair]:
        """Desaturated (B&W) input — for colorization benchmarks."""
        pairs = []
        for i, ref in enumerate(self._references()):
            gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
            degraded = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            pairs.append(FramePair(degraded=degraded, reference=ref, name=f"grayscale_{i}"))
        return pairs

    def mixed_degradation(self) -> list[FramePair]:
        """
        Combined degradation (downscale + noise + JPEG) — mirrors real-world
        'real-world SR' evaluation used in Real-ESRGAN / Blind-SR papers.
        """
        rng = np.random.default_rng(self.seed + 10)
        pairs = []
        for i, ref in enumerate(self._references()):
            # 1. Downscale
            factor = 4
            small = cv2.resize(
                ref, (self.width // factor, self.height // factor), interpolation=cv2.INTER_CUBIC
            )
            # 2. Add noise
            noise = rng.normal(0, 15, small.shape).astype(np.float32)
            small = np.clip(small.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            # 3. JPEG
            _, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 60]
            )
            small = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            # Upscale back for comparison-ready input
            degraded = cv2.resize(small, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            pairs.append(
                FramePair(
                    degraded=degraded, reference=ref, name=f"mixed_degrad_{i}", dataset="real_world"
                )
            )
        return pairs

    def standard_patterns(self, factor: int = 4) -> list[FramePair]:
        """
        Classic test-image-inspired LR→HR pairs for SR evaluation.
        Mirrors the statistical content of Set5/Set14 without copyright issues.

        Patterns: Lena-like, Cameraman-like, Baboon-like, Urban-like, Gradient.
        """
        generators = [_lena_like, _cameraman_like, _baboon_like, _urban_like]
        patterns = [g(self.width) for g in generators]
        # Fill up to num_pairs with gradient variations
        rng = np.random.default_rng(self.seed + 99)
        while len(patterns) < self.num_pairs:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for c in range(3):
                base = int(rng.integers(50, 180))
                frame[:, :, c] = np.tile(
                    np.linspace(base, base + 60, self.width, dtype=np.uint8), (self.height, 1)
                )
            patterns.append(frame)

        pairs = []
        for i, ref in enumerate(patterns[: self.num_pairs]):
            lr = cv2.resize(
                ref, (self.width // factor, self.height // factor), interpolation=cv2.INTER_CUBIC
            )
            degraded = cv2.resize(lr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            pairs.append(
                FramePair(
                    degraded=degraded,
                    reference=ref,
                    name=f"std_pattern_x{factor}_{i}",
                    scale_factor=1,
                    dataset="standard_patterns",
                )
            )
        return pairs

    def all_degradations(self) -> dict[str, list[FramePair]]:
        """Return every degradation type keyed by name."""
        return {
            "gaussian_blur": self.gaussian_blur(),
            "add_noise": self.add_noise(),
            "jpeg_compress": self.jpeg_compress(),
            "downscale_bicubic": self.downscale_bicubic(),
            "simulate_scratch": self.simulate_scratch(),
            "grayscale": self.grayscale(),
            "mixed_degradation": self.mixed_degradation(),
            "standard_patterns": self.standard_patterns(),
        }

    # ── Baseline comparison ────────────────────────────────────────────────────

    def compute_baselines(self, factor: int = 4) -> dict[str, BaselineResult]:
        """
        Compute PSNR/SSIM for classical upscaling methods on standard patterns.

        Methods compared:
          nearest    — nearest-neighbour (fastest, worst quality)
          bilinear   — bilinear interpolation
          bicubic    — bicubic interpolation (standard SR paper baseline)
          lanczos    — Lanczos-4 (best classical, often used in video players)
          edsr_proxy — sharpened bicubic approximation of CNN-SR quality range

        These are the baselines used in SRCNN, EDSR, RCAN, Real-ESRGAN papers.
        Any RestoraX restorer should comfortably exceed bicubic PSNR/SSIM.
        """
        import time

        from restorax.metrics.full_reference import psnr as compute_psnr
        from restorax.metrics.full_reference import ssim as compute_ssim

        refs = self._references()
        results = {}
        methods = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4,
        }

        for method_name, interp in methods.items():
            psnrs, ssims, times = [], [], []
            for ref in refs:
                lr = cv2.resize(
                    ref,
                    (self.width // factor, self.height // factor),
                    interpolation=cv2.INTER_CUBIC,
                )
                t0 = time.perf_counter()
                sr = cv2.resize(lr, (self.width, self.height), interpolation=interp)
                times.append(time.perf_counter() - t0)
                psnrs.append(compute_psnr(sr, ref))
                ssims.append(compute_ssim(sr, ref))

            n = len(refs)
            results[method_name] = BaselineResult(
                method=method_name,
                psnr=float(np.mean([v for v in psnrs if v < 999])),
                ssim=float(np.mean(ssims)),
                fps=n / sum(times) if sum(times) > 0 else 0.0,
            )

        # Sharpened-bicubic: approximates the quality range of early CNN methods
        kernel = np.array([[0, -0.2, 0], [-0.2, 1.8, -0.2], [0, -0.2, 0]], np.float32)
        psnrs, ssims, times = [], [], []
        for ref in refs:
            lr = cv2.resize(
                ref, (self.width // factor, self.height // factor), interpolation=cv2.INTER_CUBIC
            )
            t0 = time.perf_counter()
            bc = cv2.resize(lr, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
            sr = np.clip(cv2.filter2D(bc, -1, kernel), 0, 255).astype(np.uint8)
            times.append(time.perf_counter() - t0)
            psnrs.append(compute_psnr(sr, ref))
            ssims.append(compute_ssim(sr, ref))

        results["sharpened_bicubic"] = BaselineResult(
            method="sharpened_bicubic",
            psnr=float(np.mean([v for v in psnrs if v < 999])),
            ssim=float(np.mean(ssims)),
            fps=len(refs) / sum(times) if sum(times) > 0 else 0.0,
        )

        return results

    # ── Internal ───────────────────────────────────────────────────────────────

    def _references(self) -> list[np.ndarray]:
        """Generate num_pairs reference frames deterministically."""
        if self.use_standard_patterns:
            generators = [_lena_like, _cameraman_like, _baboon_like, _urban_like]
            frames = [g(self.width) for g in generators]
            rng = np.random.default_rng(self.seed)
            while len(frames) < self.num_pairs:
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                for c in range(3):
                    base = int(rng.integers(50, 180))
                    frame[:, :, c] = np.tile(
                        np.linspace(base, base + 60, self.width, dtype=np.uint8), (self.height, 1)
                    )
                frames.append(frame)
            return frames[: self.num_pairs]

        # Synthetic: gradient + colored blobs
        rng = np.random.default_rng(self.seed)
        refs = []
        for _ in range(self.num_pairs):
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            for c in range(3):
                base = rng.integers(30, 200)
                gradient = np.linspace(base, base + 50, self.width, dtype=np.uint8)
                frame[:, :, c] = gradient[np.newaxis, :]
            for _ in range(5):
                cx = int(rng.integers(10, self.width - 10))
                cy = int(rng.integers(10, self.height - 10))
                r = int(rng.integers(5, 20))
                color = tuple(int(v) for v in rng.integers(0, 255, 3).tolist())
                cv2.circle(frame, (cx, cy), r, color, -1)
            refs.append(frame)
        return refs
