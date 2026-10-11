#!/usr/bin/env python
"""
Generate real-world restoration samples for ALL 25 restorers.

Runs on GPU (RTX 3080). Processes real CC-BY media from samples/.
Outputs before/after PNGs/WAVs/MP4s to docs/assets/restorations/
and a manifest.json recording status per model.

Usage:
    conda run -n restorax python scripts/generate_real_samples.py
"""

import json
import sys
import time
import types
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# basicsr imports torchvision.transforms.functional_tensor which was removed
# in newer torchvision; patch before any restorax imports.
_ft = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms.functional import rgb_to_grayscale  # noqa: E402

_ft.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _ft

import contextlib

import cv2
import numpy as np
import soundfile as sf
import torch

SAMPLES = REPO / "samples"
OUT = REPO / "docs/assets/restorations"
OUT.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM"
    )

manifest: dict = {}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _free_vram() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def _load_frame(path: Path, max_px: int = 720) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)
    return img


def _extract_frame(path: Path, n: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {n} from {path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _save_comparison(
    name: str, before: np.ndarray, after: np.ndarray, original: np.ndarray | None = None
) -> None:
    cv2.imwrite(str(OUT / f"{name}_before.png"), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / f"{name}_after.png"), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    h = min(before.shape[0], after.shape[0], 540)
    panels = []
    if original is not None:
        cv2.imwrite(str(OUT / f"{name}_original.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        panels.append(cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h)))
    panels.append(cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h)))
    panels.append(cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h)))
    cv2.imwrite(
        str(OUT / f"{name}_composite.png"),
        cv2.cvtColor(np.concatenate(panels, axis=1), cv2.COLOR_RGB2BGR),
    )


def _save_video_clip(name: str, src: Path, process_fn, start: int = 30, n_frames: int = 75) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_before = cv2.VideoWriter(str(OUT / f"{name}_before.mp4"), fourcc, fps, (w, h))
    processed0 = process_fn(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))
    ph, pw = processed0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    out_after = cv2.VideoWriter(str(OUT / f"{name}_after.mp4"), fourcc, fps, (pw, ph))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = process_fn(rgb)
        ph2, pw2 = processed.shape[:2]
        out_before.write(cv2.resize(frame, (pw2, ph2)))
        out_after.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    cap.release()
    out_before.release()
    out_after.release()


def _spectrogram(name: str, before: np.ndarray, after: np.ndarray, sr: int) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mono = lambda a: a[:, 0] if a.ndim > 1 else a  # noqa: E731
        n = min(sr * 5, len(mono(before)), len(mono(after)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, sig, title in zip(
            axes, [mono(before)[:n], mono(after)[:n]], ["Input", "Restored"], strict=True
        ):
            ax.specgram(sig, Fs=sr, cmap="magma")
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(str(OUT / f"{name}_spectrogram.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"  spectrogram skipped: {e}")


def _run_video(
    key: str,
    cls_path: str,
    cls_name: str,
    sample: str,
    max_px: int = 512,
    params_kw: dict | None = None,
    grayscale_input: bool = False,
    original_max_px: int | None = None,
    save_video: bool = False,
    video_src: str | None = None,
) -> None:
    """Generic runner for any video/image restorer."""
    print(f"\n[{key}] {cls_name}")
    restorer = None
    try:
        mod = __import__(f"restorax.restorers.{cls_path}", fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        from restorax.core.restorer import RestorerParams  # noqa: PLC0415

        sample_path = SAMPLES / sample
        if sample_path.suffix in (".mp4", ".mkv", ".avi", ".mov"):
            frame = _extract_frame(sample_path, n=30)
        else:
            frame = _load_frame(sample_path, max_px=max_px)

        original = None
        if original_max_px and sample_path.suffix not in (".mp4", ".mkv"):
            original = _load_frame(sample_path, max_px=original_max_px)

        if grayscale_input:
            frame = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        restorer = cls()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_frame(frame, RestorerParams(**(params_kw or {})))
        elapsed = time.time() - t0
        print(f"  {frame.shape} → {out.shape}  {elapsed:.1f}s")
        _save_comparison(key, frame, out, original=original)

        if save_video and video_src:
            _save_video_clip(
                key,
                SAMPLES / video_src,
                lambda f: restorer.process_frame(f, RestorerParams(**(params_kw or {}))),
                start=30,
                n_frames=75,
            )

        manifest[key] = {"status": "real", "model": cls_name, "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest[key] = {"status": "failed", "error": str(e)[:200]}
    finally:
        if restorer is not None:
            with contextlib.suppress(Exception):
                restorer.unload()
        _free_vram()


def _run_audio(key: str, cls_path: str, cls_name: str, sample: str) -> None:
    """Generic runner for any audio restorer."""
    print(f"\n[{key}] {cls_name}")
    restorer = None
    try:
        mod = __import__(f"restorax.restorers.{cls_path}", fromlist=[cls_name])
        cls = getattr(mod, cls_name)
        from restorax.audio.pipeline import AudioRestorerParams  # noqa: PLC0415

        audio, sr = sf.read(str(SAMPLES / sample), dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        print(f"  input: {audio.shape}, sr={sr}")

        restorer = cls()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_audio(audio, AudioRestorerParams(sample_rate=sr))
        elapsed = time.time() - t0
        print(f"  output: {out.shape}  {elapsed:.1f}s")

        sf.write(str(OUT / f"{key}_before.wav"), audio, sr)
        sf.write(str(OUT / f"{key}_after.wav"), out, sr)
        _spectrogram(key, audio, out, sr)
        manifest[key] = {"status": "real", "model": cls_name, "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest[key] = {"status": "failed", "error": str(e)[:200]}
    finally:
        if restorer is not None:
            with contextlib.suppress(Exception):
                restorer.unload()
        _free_vram()


# ── Pipeline: all 25 restorers ─────────────────────────────────────────────────
#
# Video: (key, module_path, class_name, sample_file, max_px, params_kw,
#         grayscale_input, original_max_px, save_video, video_src)
# Audio: (key, module_path, class_name, sample_file)

VIDEO_JOBS = [
    # ── Super-resolution (10) ──────────────────────────────────────────────────
    (
        "sr_real_esrgan",
        "super_resolution.real_esrgan",
        "RealESRGANx4Restorer",
        "photo/portrait_tearsofsteel.png",
        256,
        {"scale": 4, "half_precision": False},
        False,
        1024,
        False,
        None,
    ),
    (
        "sr_basicvsr_pp",
        "super_resolution.basicvsr_pp",
        "BasicVSRPlusPlusRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_upscale_a_video",
        "super_resolution.upscale_a_video",
        "UpscaleAVideoRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_vrt",
        "super_resolution.vrt",
        "VRTRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_mamba_ir",
        "super_resolution.mamba_ir",
        "MambaIRRestorer",
        "photo/portrait_tearsofsteel.png",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_tdm",
        "super_resolution.tdm",
        "TDMRestorer",
        "photo/portrait_tearsofsteel.png",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_seedvr",
        "super_resolution.seedvr",
        "SeedVRRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_waifu2x",
        "super_resolution.waifu2x",
        "Waifu2xRestorer",
        "photo/scene_sintel.png",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_flashvsr",
        "super_resolution.flashvsr",
        "FlashVSRRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "sr_evtexture",
        "super_resolution.evtexture",
        "EvTextureRestorer",
        "video/film_sintel.mp4",
        360,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── Face restoration (4) ───────────────────────────────────────────────────
    (
        "face_codeformer",
        "face_restoration.codeformer",
        "CodeFormerRestorer",
        "photo/portrait_tearsofsteel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "face_codeformer_pp",
        "face_restoration.codeformer_pp",
        "CodeFormerPlusPlusRestorer",
        "photo/portrait_tearsofsteel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "face_gfpgan",
        "face_restoration.gfpgan",
        "GFPGANRestorer",
        "photo/portrait_tearsofsteel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "face_dicface",
        "face_restoration.dicface",
        "DicFaceRestorer",
        "photo/portrait_tearsofsteel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── Colorization (1) ──────────────────────────────────────────────────────
    (
        "colorization_ddcolor",
        "colorization.ddcolor",
        "DDColorRestorer",
        "photo/scene_sintel.png",
        512,
        {},
        True,
        None,
        False,
        None,
    ),
    # ── Frame interpolation (1) ───────────────────────────────────────────────
    (
        "interp_rife",
        "frame_interpolation.rife",
        "RIFERestorer",
        "video/film_sintel.mp4",
        480,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── Artifact removal (1) ──────────────────────────────────────────────────
    (
        "artifact_scratch",
        "artifact_removal.scratch_removal",
        "ScratchRemovalRestorer",
        "photo/portrait_tearsofsteel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── HDR (1) ───────────────────────────────────────────────────────────────
    (
        "hdr_tvdm",
        "hdr.hdrtvdm",
        "HDRTVDMRestorer",
        "photo/scene_sintel.png",
        512,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── Stabilization (2) ─────────────────────────────────────────────────────
    (
        "stab_deepflow",
        "stabilization.deep_flow_stab",
        "VideoStabilizationRestorer",
        "video/film_sintel.mp4",
        480,
        {},
        False,
        None,
        False,
        None,
    ),
    (
        "stab_gavs",
        "stabilization.gavs",
        "GaVSRestorer",
        "video/film_sintel.mp4",
        480,
        {},
        False,
        None,
        False,
        None,
    ),
    # ── Deinterlacing (2) ─────────────────────────────────────────────────────
    (
        "deint_ai",
        "deinterlacing.ai_deinterlace",
        "AIDeinterlaceRestorer",
        "video/film_sintel.mp4",
        480,
        {},
        False,
        None,
        True,
        "video/film_sintel.mp4",
    ),
    (
        "deint_yadif",
        "deinterlacing.yadif_deinterlace",
        "YadifDeinterlaceRestorer",
        "video/film_sintel.mp4",
        480,
        {},
        False,
        None,
        True,
        "video/film_sintel.mp4",
    ),
]

AUDIO_JOBS = [
    ("audio_demucs", "audio.demucs", "DemucsRestorer", "audio/music_bigbuckbunny.wav"),
    ("audio_voicefixer", "audio.voicefixer", "VoiceFixerRestorer", "audio/speech_tearsofsteel.wav"),
    ("audio_rnnoise", "audio.rnnoise", "RNNoiseRestorer", "audio/speech_tearsofsteel.wav"),
]


if __name__ == "__main__":
    t_start = time.time()

    for args in VIDEO_JOBS:
        key, mod, cls, sample, max_px, params_kw, gray, orig_px, save_vid, vid_src = args
        try:
            _run_video(key, mod, cls, sample, max_px, params_kw, gray, orig_px, save_vid, vid_src)
        except Exception as e:
            print(f"  UNCAUGHT in {key}: {e}", file=sys.stderr)

    for args in AUDIO_JOBS:
        key, mod, cls, sample = args
        try:
            _run_audio(key, mod, cls, sample)
        except Exception as e:
            print(f"  UNCAUGHT in {key}: {e}", file=sys.stderr)

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    elapsed_total = time.time() - t_start
    real = sum(1 for v in manifest.values() if v.get("status") == "real")
    failed = sum(1 for v in manifest.values() if v.get("status") == "failed")
    print(f"\nDone in {elapsed_total:.0f}s — Real: {real}/{len(manifest)}, Failed: {failed}")
    print(f"Manifest: {manifest_path}")
