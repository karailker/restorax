#!/usr/bin/env python
"""
Generate real-world restoration samples using actual AI model weights.

Runs on GPU (RTX 3080). Processes real CC-BY media from samples/.
Outputs before/after PNGs to docs/assets/restorations/ and a manifest.json
recording which models ran with real weights vs. fell back.

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
# in newer torchvision; patch it before any restorax imports.
_ft = types.ModuleType("torchvision.transforms.functional_tensor")
from torchvision.transforms.functional import rgb_to_grayscale  # noqa: E402
_ft.rgb_to_grayscale = rgb_to_grayscale
sys.modules["torchvision.transforms.functional_tensor"] = _ft

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
    print(f"GPU: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).total_memory // 1024**3} GB VRAM")

manifest: dict = {}


def _save_comparison(name: str, before: np.ndarray, after: np.ndarray) -> None:
    cv2.imwrite(str(OUT / f"{name}_before.png"), cv2.cvtColor(before, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(OUT / f"{name}_after.png"), cv2.cvtColor(after, cv2.COLOR_RGB2BGR))
    h = min(before.shape[0], after.shape[0], 720)
    b_h = cv2.resize(before, (int(before.shape[1] * h / before.shape[0]), h))
    a_h = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h))
    cv2.imwrite(str(OUT / f"{name}_composite.png"),
                cv2.cvtColor(np.concatenate([b_h, a_h], axis=1), cv2.COLOR_RGB2BGR))
    print(f"  saved {name}_before/after/composite.png")


def _free_vram() -> None:
    torch.cuda.empty_cache()
    if DEVICE.type == "cuda":
        print(f"  VRAM free: {torch.cuda.mem_get_info()[0] // 1024**2} MB")


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


def _save_video_clip(name: str, src: Path, process_fn, start: int = 30,
                     n_frames: int = 75) -> None:
    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_before = cv2.VideoWriter(str(OUT / f"{name}_before.mp4"), fourcc, fps, (w, h))
    out_after = cv2.VideoWriter(str(OUT / f"{name}_after.mp4"), fourcc, fps, (w, h))
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed = process_fn(rgb)
        ph, pw = processed.shape[:2]
        out_before.write(cv2.resize(frame, (pw, ph)))
        out_after.write(cv2.cvtColor(processed, cv2.COLOR_RGB2BGR))
    cap.release()
    out_before.release()
    out_after.release()
    print(f"  saved {name}_before/after.mp4 ({n_frames} frames @ {fps:.0f} fps)")


def _spectrogram(name: str, before: np.ndarray, after: np.ndarray, sr: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mono = lambda a: a[:, 0] if a.ndim > 1 else a  # noqa: E731
        n = min(sr * 5, len(mono(before)), len(mono(after)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, sig, title in zip(axes, [mono(before)[:n], mono(after)[:n]],
                                   ["Input", "Restored"]):
            ax.specgram(sig, Fs=sr, cmap="magma")
            ax.set_title(title)
        plt.tight_layout()
        plt.savefig(str(OUT / f"{name}_spectrogram.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"  spectrogram skipped: {e}")


# ── Per-restorer runners ──────────────────────────────────────────────────────

def run_super_resolution() -> None:
    print("\n[SR] Real-ESRGAN x4 on portrait_tearsofsteel.png")
    restorer = None
    try:
        from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
        from restorax.core.restorer import RestorerParams
        frame = _load_frame(SAMPLES / "photo/portrait_tearsofsteel.png", max_px=256)
        print(f"  input: {frame.shape}")
        restorer = RealESRGANx4Restorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_frame(frame, RestorerParams(scale=4, half_precision=False))
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        _save_comparison("sr_real_esrgan", frame, out)
        manifest["sr_real_esrgan"] = {"status": "real", "model": "RealESRGAN_x4plus",
                                       "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["sr_real_esrgan"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


def run_face_restoration() -> None:
    print("\n[FACE] CodeFormer on portrait_tearsofsteel.png")
    restorer = None
    try:
        from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
        from restorax.core.restorer import RestorerParams
        frame = _load_frame(SAMPLES / "photo/portrait_tearsofsteel.png", max_px=512)
        restorer = CodeFormerRestorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_frame(frame, RestorerParams())
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        _save_comparison("face_codeformer", frame, out)
        manifest["face_codeformer"] = {"status": "real", "model": "CodeFormer",
                                        "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["face_codeformer"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


def run_colorization() -> None:
    print("\n[COLOR] DDColor on scene_sintel.png (grayscale input)")
    restorer = None
    try:
        from restorax.restorers.colorization.ddcolor import DDColorRestorer
        from restorax.core.restorer import RestorerParams
        frame = _load_frame(SAMPLES / "photo/scene_sintel.png", max_px=512)
        bw = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        restorer = DDColorRestorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_frame(bw, RestorerParams())
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        _save_comparison("colorization_ddcolor", bw, out)
        manifest["colorization_ddcolor"] = {"status": "real", "model": "DDColor",
                                              "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["colorization_ddcolor"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


def run_deinterlacing() -> None:
    print("\n[DEINT] YADIF on film_sintel.mp4")
    restorer = None
    try:
        from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer
        from restorax.core.restorer import RestorerParams
        video_path = SAMPLES / "video/film_sintel.mp4"
        frame = _extract_frame(video_path, n=30)
        restorer = YadifDeinterlaceRestorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_frame(frame, RestorerParams())
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        _save_comparison("deinterlace_yadif", frame, out)
        # Also save a real video clip (3 seconds / 75 frames)
        _save_video_clip("deinterlace_yadif",
                         video_path,
                         lambda f: restorer.process_frame(f, RestorerParams()),
                         start=30, n_frames=75)
        manifest["deinterlace_yadif"] = {"status": "real", "model": "YADIF (classical)",
                                          "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["deinterlace_yadif"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


def run_audio_demucs() -> None:
    print("\n[AUDIO] Demucs on music_bigbuckbunny.wav")
    restorer = None
    try:
        from restorax.restorers.audio.demucs import DemucsRestorer
        from restorax.audio.pipeline import AudioRestorerParams
        audio, sr = sf.read(str(SAMPLES / "audio/music_bigbuckbunny.wav"), dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        print(f"  input: {audio.shape}, sr={sr}")
        restorer = DemucsRestorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_audio(audio, AudioRestorerParams(sample_rate=sr))
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        sf.write(str(OUT / "audio_demucs_before.wav"), audio, sr)
        sf.write(str(OUT / "audio_demucs_after.wav"), out, sr)
        _spectrogram("audio_demucs", audio, out, sr)
        manifest["audio_demucs"] = {"status": "real", "model": "Demucs htdemucs",
                                     "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["audio_demucs"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


def run_audio_voicefixer() -> None:
    print("\n[AUDIO] VoiceFixer on speech_tearsofsteel.wav")
    restorer = None
    try:
        from restorax.restorers.audio.voicefixer import VoiceFixerRestorer
        from restorax.audio.pipeline import AudioRestorerParams
        audio, sr = sf.read(str(SAMPLES / "audio/speech_tearsofsteel.wav"), dtype="float32")
        if audio.ndim == 1:
            audio = audio[:, np.newaxis]
        print(f"  input: {audio.shape}, sr={sr}")
        restorer = VoiceFixerRestorer()
        t0 = time.time()
        restorer.load(DEVICE)
        out = restorer.process_audio(audio, AudioRestorerParams(sample_rate=sr))
        elapsed = time.time() - t0
        print(f"  output: {out.shape}, {elapsed:.1f}s")
        sf.write(str(OUT / "audio_voicefixer_before.wav"), audio, sr)
        sf.write(str(OUT / "audio_voicefixer_after.wav"), out, sr)
        _spectrogram("audio_voicefixer", audio, out, sr)
        manifest["audio_voicefixer"] = {"status": "real", "model": "VoiceFixer",
                                         "elapsed_s": round(elapsed, 2)}
    except Exception as e:
        print(f"  FAILED: {e}")
        manifest["audio_voicefixer"] = {"status": "failed", "error": str(e)}
    finally:
        if restorer is not None:
            restorer.unload()
        _free_vram()


# ── Main ──────────────────────────────────────────────────────────────────────

PIPELINE = [
    run_super_resolution,
    run_face_restoration,
    run_colorization,
    run_deinterlacing,
    run_audio_demucs,
    run_audio_voicefixer,
]

if __name__ == "__main__":
    t_start = time.time()
    for fn in PIPELINE:
        try:
            fn()
        except Exception as e:
            print(f"  UNCAUGHT in {fn.__name__}: {e}", file=sys.stderr)

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    elapsed_total = time.time() - t_start
    real = sum(1 for v in manifest.values() if v.get("status") == "real")
    failed = sum(1 for v in manifest.values() if v.get("status") == "failed")
    print(f"\nDone in {elapsed_total:.0f}s. Real weights: {real}/{len(manifest)}, "
          f"Failed: {failed}")
    print(f"Manifest: {manifest_path}")
