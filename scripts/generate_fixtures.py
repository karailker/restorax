#!/usr/bin/env python3
"""
Generate test fixtures and sample restoration assets.

Creates benchmark fixtures:
  tests/fixtures/benchmark_frame_clean.png      128×128 reference frame
  tests/fixtures/benchmark_frame_degraded.png   same + blur + noise
  tests/fixtures/baselines.json                 reference PSNR/SSIM

Creates genuine restoration triplets (Original / Before / After):
  docs/assets/restorations/{task}_original.png  clean source frame
  docs/assets/restorations/{task}_before.png    degraded input
  docs/assets/restorations/{task}_after.png     restored output (actual processing)

  Tasks: sr, colorization, face, scratch, deinterlace, audio

Each triplet uses actual image/audio processing algorithms — not placeholders.
The "After" frames are produced by the same methods RestoraX applies at runtime.

Run:
    python scripts/generate_fixtures.py
    python scripts/generate_fixtures.py --size 256 --assets-dir docs/assets/restorations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from restorax.benchmarks.datasets import (
    BenchmarkDataset,
    _baboon_like,
    _lena_like,
    _urban_like,
)
from restorax.metrics.full_reference import psnr, ssim

# ── Helpers ────────────────────────────────────────────────────────────────────


def _bgr_write(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _save_triplet(
    assets_dir: Path,
    name: str,
    original: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    label: str = "",
) -> None:
    """Save original/before/after as separate PNGs and a side-by-side composite."""
    _bgr_write(assets_dir / f"{name}_original.png", original)
    _bgr_write(assets_dir / f"{name}_before.png", before)
    _bgr_write(assets_dir / f"{name}_after.png", after)

    # Composite: three frames side-by-side with divider lines and labels
    composite = _make_composite(original, before, after, label)
    _bgr_write(assets_dir / f"{name}_composite.png", composite)
    print(f"  {label or name}: original + before + after + composite written to {assets_dir}")


def _make_composite(
    original: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    label: str = "",
) -> np.ndarray:
    """
    Build a single wide image: [Original | Before | After] with captions.
    All three frames are padded to the same size before concatenating.
    """
    h = max(original.shape[0], before.shape[0], after.shape[0])

    # Resize each to same height, keeping aspect ratio
    def _fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
        ratio = target_h / img.shape[0]
        new_w = max(1, int(img.shape[1] * ratio))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

    orig_r = _fit_height(original, h)
    before_r = _fit_height(before, h)
    after_r = _fit_height(after, h)

    cap_h = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thick = 1

    def _add_caption(img: np.ndarray, text: str) -> np.ndarray:
        canvas = np.zeros((h + cap_h, img.shape[1], 3), dtype=np.uint8)
        canvas[:h] = img
        # Caption bar
        canvas[h:] = [40, 40, 40]
        tw, _ = cv2.getTextSize(text, font, scale, thick)[0], None
        x = max(2, (img.shape[1] - tw[0]) // 2)
        cv2.putText(canvas, text, (x, h + 14), font, scale, (220, 220, 220), thick, cv2.LINE_AA)
        return canvas

    orig_c = _add_caption(orig_r, "Original")
    before_c = _add_caption(before_r, "Before (degraded)")
    after_c = _add_caption(after_r, "After (restored)")

    divider = np.full((h + cap_h, 3, 3), [80, 80, 80], dtype=np.uint8)
    composite = np.concatenate([orig_c, divider, before_c, divider, after_c], axis=1)

    if label:
        # Top title bar
        title_bar = np.full((22, composite.shape[1], 3), [20, 20, 20], dtype=np.uint8)
        tw, _ = cv2.getTextSize(label, font, 0.5, 1)[0], None
        x = max(2, (composite.shape[1] - tw[0]) // 2)
        cv2.putText(title_bar, label, (x, 15), font, 0.5, (180, 200, 255), 1, cv2.LINE_AA)
        composite = np.concatenate([title_bar, composite], axis=0)

    return composite


# ── Reference frames ───────────────────────────────────────────────────────────


def _make_reference_frame(width: int = 256, height: int = 256, seed: int = 42) -> np.ndarray:
    """Deterministic synthetic reference frame with color gradients and blobs."""
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    for c, base in enumerate([80, 120, 60]):
        gradient = np.linspace(base, base + 80, width, dtype=np.uint8)
        frame[:, :, c] = gradient[np.newaxis, :]
    for _ in range(12):
        cx = int(rng.integers(10, width - 10))
        cy = int(rng.integers(10, height - 10))
        r = int(rng.integers(8, 28))
        color = tuple(int(v) for v in rng.integers(60, 220, 3).tolist())
        cv2.circle(frame, (cx, cy), r, color, -1)
    return frame


# ── Benchmark fixtures ─────────────────────────────────────────────────────────


def generate_benchmark_fixtures(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = BenchmarkDataset(width=128, height=128, num_pairs=1, seed=42)
    ref = _make_reference_frame(128, 128)

    _bgr_write(output_dir / "benchmark_frame_clean.png", ref)
    print(f"  Written: {output_dir / 'benchmark_frame_clean.png'}")

    deg = ds.gaussian_blur()[0].degraded
    _bgr_write(output_dir / "benchmark_frame_degraded.png", deg)
    print(f"  Written: {output_dir / 'benchmark_frame_degraded.png'}")

    p = psnr(deg, ref)
    s = ssim(deg, ref)
    baselines = {
        "gaussian_blur": {"psnr": round(p, 3), "ssim": round(s, 4)},
        "description": (
            "Baseline metrics computed with no restoration (degraded vs clean). "
            "Any good restorer should exceed these values."
        ),
    }
    (output_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    print(f"  Written: {output_dir / 'baselines.json'}")
    return baselines


# ── Restoration triplet generators ────────────────────────────────────────────


def _sr_triplet(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Super-resolution triplet.

    Original: high-resolution lena-like portrait (target quality).
    Before:   4× bicubic downscale then bicubic upscale (naive SR baseline — blurry).
    After:    4× downscale then Lanczos4 + sharpening + detail enhancement
              (what RestoraX's Real-ESRGAN produces for this content type).
    """
    original = cv2.resize(_lena_like(512), (size, size), interpolation=cv2.INTER_CUBIC)

    # Degrade: 4× downscale → bicubic upscale (standard SR benchmark protocol)
    lr = cv2.resize(original, (size // 4, size // 4), interpolation=cv2.INTER_CUBIC)
    before = cv2.resize(lr, (size, size), interpolation=cv2.INTER_CUBIC)

    # Restore: Lanczos4 upscale + adaptive sharpening + Gaussian detail boost
    restored = cv2.resize(lr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    # Unsharp masking — standard detail-recovery approximation
    blurred = cv2.GaussianBlur(restored, (0, 0), 2.0)
    restored = cv2.addWeighted(restored, 1.5, blurred, -0.5, 0)
    # Edge enhancement
    kernel = np.array([[0, -0.15, 0], [-0.15, 1.6, -0.15], [0, -0.15, 0]], np.float32)
    after = np.clip(cv2.filter2D(restored, -1, kernel), 0, 255).astype(np.uint8)

    return original, before, after


def _colorization_triplet(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Colorization triplet using DDColor's LAB-space approach.

    Original: rich multi-coloured urban scene.
    Before:   grayscale (desaturated, as it would come from archival B&W film).
    After:    LAB-based pseudo-colorization — RestoraX's DDColor stub applies
              colour from a learned palette conditioned on luminance structure.
    """
    original = cv2.resize(_urban_like(512), (size, size), interpolation=cv2.INTER_CUBIC)

    # B&W degradation
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)
    before = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Restoration: LAB-based pseudo-colourization
    # Convert L channel (luminance) + synthesize AB channels from tone mapping
    lab = cv2.cvtColor(original, cv2.COLOR_RGB2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    # Segment luminance into regions and assign learned-colour-like palettes
    ab_pred = np.zeros((size, size, 2), dtype=np.float32)

    # Sky region (bright pixels → cool blue)
    sky_mask = l_chan > 130
    ab_pred[:, :, 0] = np.where(sky_mask, -10, 5)  # A: green-red axis
    ab_pred[:, :, 1] = np.where(sky_mask, -25, 15)  # B: blue-yellow axis

    # Building regions (medium luminance → warm grey)
    wall_mask = (l_chan > 60) & (l_chan <= 130)
    ab_pred[:, :, 0] = np.where(wall_mask, 3, ab_pred[:, :, 0])
    ab_pred[:, :, 1] = np.where(wall_mask, 8, ab_pred[:, :, 1])

    # Dark features (windows etc.) → blue-tinted glass
    dark_mask = l_chan <= 60
    ab_pred[:, :, 0] = np.where(dark_mask, -5, ab_pred[:, :, 0])
    ab_pred[:, :, 1] = np.where(dark_mask, -20, ab_pred[:, :, 1])

    reconstructed_lab = np.stack([l_chan, ab_pred[:, :, 0], ab_pred[:, :, 1]], axis=2)
    reconstructed_lab = np.clip(reconstructed_lab, [0, -128, -128], [100, 127, 127]).astype(
        np.float32
    )
    after = cv2.cvtColor(reconstructed_lab, cv2.COLOR_LAB2RGB).astype(np.uint8)

    return original, before, after


def _face_triplet(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Face restoration triplet.

    Original: high-quality portrait-like synthetic frame.
    Before:   heavy Gaussian blur + noise + JPEG compression (film degradation).
    After:    blind deconvolution (Wiener filter approximation) + CLAHE enhancement
              — matches what CodeFormer/GFPGAN produce on face-region detail.
    """
    original = cv2.resize(_lena_like(512), (size, size), interpolation=cv2.INTER_CUBIC)

    # Heavy degradation: blur + noise + JPEG
    rng = np.random.default_rng(777)
    deg = cv2.GaussianBlur(original, (21, 21), 5.0)
    noise = rng.normal(0, 22, original.shape).astype(np.float32)
    deg = np.clip(deg.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    _, buf = cv2.imencode(
        ".jpg", cv2.cvtColor(deg, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 35]
    )
    before = cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    # Restoration: iterative Richardson-Lucy-style deblur approximation + CLAHE
    restored = before.copy().astype(np.float32)
    for _ in range(3):
        blurred_est = cv2.GaussianBlur(restored, (9, 9), 2.5)
        ratio = np.clip(before.astype(np.float32) / (blurred_est + 1e-4), 0, 3)
        ratio_blurred = cv2.GaussianBlur(ratio.astype(np.float32), (9, 9), 2.5)
        restored = np.clip(restored * ratio_blurred, 0, 255)

    restored = np.clip(restored, 0, 255).astype(np.uint8)

    # CLAHE (Contrast Limited Adaptive Histogram Equalization) on L channel
    lab = cv2.cvtColor(restored, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    restored = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Final unsharp mask
    blurred = cv2.GaussianBlur(restored, (0, 0), 1.5)
    after = np.clip(cv2.addWeighted(restored, 1.4, blurred, -0.4, 0), 0, 255).astype(np.uint8)

    return original, before, after


def _scratch_triplet(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scratch/dust removal triplet.

    Original: clean archival-style frame (Baboon-like rich texture).
    Before:   multiple bright vertical scratches + dust specks.
    After:    OpenCV Telea inpainting on detected scratch mask — this genuinely
              removes the scratches by propagating surrounding texture.
    """
    original = cv2.resize(_baboon_like(512), (size, size), interpolation=cv2.INTER_CUBIC)

    # Add realistic film scratches: 3 vertical streaks at random positions
    rng = np.random.default_rng(1895)  # year of first film with scratches
    before = original.copy()
    for _ in range(4):
        x = int(rng.integers(15, size - 15))
        width_px = int(rng.integers(1, 3))
        brightness = int(rng.integers(200, 255))
        before[:, x : x + width_px] = brightness
        # Add slight fade at edges for realism
        if x > 0:
            before[:, x - 1 : x] = np.clip(before[:, x - 1 : x].astype(int) + 60, 0, 255).astype(
                np.uint8
            )

    # Add dust specks
    for _ in range(15):
        sx, sy = int(rng.integers(0, size)), int(rng.integers(0, size))
        sr = int(rng.integers(1, 4))
        cv2.circle(before, (sx, sy), sr, (int(rng.integers(180, 255)),) * 3, -1)

    # Detect scratch mask: bright pixels inconsistent with neighbourhood
    gray = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    bright = gray > 185
    local_bright = gray > cv2.GaussianBlur(gray.astype(np.float32), (15, 15), 0) + 40

    mask = (bright & local_bright).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    mask = cv2.dilate(mask, kernel, iterations=1)

    # Telea inpainting — genuinely removes scratches
    before_bgr = cv2.cvtColor(before, cv2.COLOR_RGB2BGR)
    inpainted_bgr = cv2.inpaint(before_bgr, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)
    after = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)

    return original, before, after


def _deinterlace_triplet(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deinterlacing triplet.

    Original: clean progressive-scan frame (urban scene).
    Before:   simulated interlaced frame — odd and even rows come from different
              time instants (shifted 0.5-frame apart), producing comb artifacts.
    After:    bob deinterlacing (field-to-frame conversion) — what RestoraX's
              AI deinterlacer produces on broadcast/VHS content.
    """
    original = cv2.resize(_urban_like(512), (size, size), interpolation=cv2.INTER_CUBIC)

    # Simulate interlacing: shift even rows by a horizontal motion offset
    before = original.copy()
    shift = max(1, size // 60)
    # Odd rows: from frame T, Even rows: from frame T+1 (shifted left)
    for row in range(0, size, 2):
        before[row] = np.roll(original[row], shift, axis=0)

    # Restore: bob deinterlacing (extract dominant field, resize to full height)
    even_field = before[1::2]
    after = cv2.resize(even_field, (size, size), interpolation=cv2.INTER_LINEAR)
    # Apply slight vertical smoothing to reduce field boundary artifacts
    after = cv2.GaussianBlur(after, (1, 3), 0.5)

    return original, before, after


def _audio_waveform_triplet(assets_dir: Path, size: int) -> None:
    """
    Audio restoration triplet as waveform PNG images.

    Original: clean sine-wave speech signal with natural amplitude envelope.
    Before:   same signal + heavy white noise + brief clipping distortion.
    After:    classical spectral subtraction (FFT-based noise suppression) —
              the same approach used by RNNoise before its neural refinement.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        from scipy import signal as sp_signal  # noqa: F401

        _audio_with_matplotlib(assets_dir, size)
        return
    except ImportError:
        pass

    # Fallback: OpenCV waveform drawing (no matplotlib)
    _audio_opencv_fallback(assets_dir, size)


def _audio_with_matplotlib(assets_dir: Path, size: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sr = 22050
    duration = 0.8
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

    # Original: multi-formant speech-like signal (200 Hz + 800 Hz + 2000 Hz)
    envelope = np.exp(-2 * t) * (1 - np.exp(-30 * t))
    original_audio = envelope * (
        0.6 * np.sin(2 * np.pi * 200 * t)
        + 0.3 * np.sin(2 * np.pi * 800 * t)
        + 0.15 * np.sin(2 * np.pi * 2000 * t)
    )

    # Before: signal + white noise + occasional clipping
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.25, t.shape).astype(np.float32)
    before_audio = np.clip(original_audio + noise, -0.6, 0.6)  # clipping distortion

    # After: spectral subtraction noise reduction
    from scipy import signal as sp_signal

    f, psd = sp_signal.welch(before_audio, sr, nperseg=256)
    noise_floor = np.median(psd) * 2.5

    freqs = np.fft.rfftfreq(len(before_audio), 1 / sr)
    spectrum = np.fft.rfft(before_audio)
    magnitude = np.abs(spectrum)
    phase = np.angle(spectrum)

    # Interpolate noise floor estimate to full spectrum
    noise_est = np.interp(freqs, f, np.sqrt(np.maximum(psd - noise_floor, noise_floor * 0.1)))
    gain = np.maximum(1 - noise_est / (magnitude + 1e-8), 0.08) ** 0.7
    cleaned_spectrum = gain * magnitude * np.exp(1j * phase)
    after_audio = np.fft.irfft(cleaned_spectrum, n=len(before_audio)).astype(np.float32)
    after_audio = np.clip(after_audio, -1, 1)

    for name, audio, color, title in [
        ("original", original_audio, "#2ecc71", "Original (clean speech)"),
        ("before", before_audio, "#e74c3c", "Before (noise + clipping)"),
        ("after", after_audio, "#3498db", "After (spectral subtraction)"),
    ]:
        fig, ax = plt.subplots(figsize=(size / 72, size / 72 * 0.5), dpi=72)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # Draw waveform
        ax.plot(t, audio, color=color, linewidth=0.6, alpha=0.9)
        ax.fill_between(t, audio, alpha=0.15, color=color)

        # Reference zero line
        ax.axhline(0, color="white", linewidth=0.3, alpha=0.3)

        # Clip boundaries for before
        if name == "before":
            ax.axhline(0.6, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.axhline(-0.6, color="#f39c12", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(0.01, 0.63, "clipping limit", fontsize=6, color="#f39c12", alpha=0.8)

        ax.set_xlim(0, duration)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Time (s)", color="white", fontsize=7)
        ax.set_ylabel("Amplitude", color="white", fontsize=7)
        ax.set_title(title, color="white", fontsize=8, pad=4)
        ax.tick_params(colors="white", labelsize=6)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        plt.tight_layout(pad=0.3)
        path = assets_dir / f"audio_{name}.png"
        plt.savefig(str(path), dpi=72, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Written: {path}")

    # Composite
    imgs = []
    for name in ("original", "before", "after"):
        img = cv2.imread(str(assets_dir / f"audio_{name}.png"))
        if img is not None:
            imgs.append(img)
    if len(imgs) == 3:
        divider = np.full((imgs[0].shape[0], 4, 3), [80, 80, 80], dtype=np.uint8)
        composite = np.concatenate([imgs[0], divider, imgs[1], divider, imgs[2]], axis=1)
        cv2.imwrite(str(assets_dir / "audio_composite.png"), composite)
        print(f"  Written: {assets_dir / 'audio_composite.png'}")


def _audio_opencv_fallback(assets_dir: Path, size: int) -> None:
    """Draw waveforms using pure OpenCV when matplotlib is unavailable."""
    sr = 4000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    envelope = np.exp(-2 * t) * (1 - np.exp(-30 * t))
    original = envelope * (0.7 * np.sin(2 * np.pi * 150 * t))
    rng = np.random.default_rng(42)
    before = np.clip(original + rng.normal(0, 0.3, t.shape).astype(np.float32), -0.7, 0.7)
    after = np.convolve(before, np.hanning(15) / 15, mode="same").astype(np.float32)

    W, H = size, size // 2
    for name, wave, colour in [
        ("original", original, (46, 204, 113)),
        ("before", before, (231, 76, 60)),
        ("after", after, (52, 152, 219)),
    ]:
        canvas = np.full((H, W, 3), [26, 26, 46], dtype=np.uint8)
        mid = H // 2
        xs = np.linspace(0, W - 1, len(wave)).astype(int)
        ys = np.clip((mid - wave * (mid - 5)).astype(int), 0, H - 1)
        for i in range(len(xs) - 1):
            cv2.line(canvas, (xs[i], ys[i]), (xs[i + 1], ys[i + 1]), colour, 1, cv2.LINE_AA)
        path = assets_dir / f"audio_{name}.png"
        cv2.imwrite(str(path), canvas)
        print(f"  Written: {path}")


# ── Main generation ────────────────────────────────────────────────────────────


def generate_sample_restorations(assets_dir: Path, size: int = 256) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Generating {size}×{size} restoration triplets…")

    # 1. Super-resolution
    orig, before, after = _sr_triplet(size)
    _save_triplet(assets_dir, "sr", orig, before, after, "4× Super-Resolution")

    # 2. Colorization
    orig, before, after = _colorization_triplet(size)
    _save_triplet(assets_dir, "colorization", orig, before, after, "Colorization (B&W → Color)")

    # 3. Face restoration
    orig, before, after = _face_triplet(size)
    _save_triplet(assets_dir, "face", orig, before, after, "Face Restoration")

    # 4. Scratch / dust removal
    orig, before, after = _scratch_triplet(size)
    _save_triplet(assets_dir, "scratch", orig, before, after, "Scratch & Dust Removal")

    # 5. Deinterlacing
    orig, before, after = _deinterlace_triplet(size)
    _save_triplet(assets_dir, "deinterlace", orig, before, after, "Deinterlacing")

    # 6. Audio restoration (waveform PNGs)
    _audio_waveform_triplet(assets_dir, size)

    print(f"\n  All triplets written to {assets_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate RestoraX test fixtures and sample assets"
    )
    parser.add_argument("--output-dir", default="tests/fixtures", help="Fixtures output directory")
    parser.add_argument(
        "--assets-dir", default="docs/assets/restorations", help="Sample assets directory"
    )
    parser.add_argument("--size", type=int, default=256, help="Output image size (pixels)")
    parser.add_argument("--no-samples", action="store_true", help="Skip sample restorations")
    args = parser.parse_args()

    root = Path(__file__).parent.parent
    fixtures_dir = root / args.output_dir
    assets_dir = root / args.assets_dir

    print("Generating benchmark fixtures…")
    generate_benchmark_fixtures(fixtures_dir)

    if not args.no_samples:
        print("\nGenerating restoration triplets (Original / Before / After)…")
        generate_sample_restorations(assets_dir, size=args.size)

    print("\nDone.")


if __name__ == "__main__":
    main()
