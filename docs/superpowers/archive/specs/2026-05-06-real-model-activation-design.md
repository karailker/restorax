# Real Model Activation — Design Spec (Track 2)

**Date:** 2026-05-06
**Track:** 2 of 5
**Status:** Approved

---

## Goal

Activate all 28 RestoraX restorers (25 video + 3 audio) for real inference. Currently only RIFE runs
real weights; every other model falls back to a silent stub. After this track:

- All non-diffusion models auto-download weights on first `load()` from HuggingFace.
- Three diffusion models (SeedVR, TDM, UpscaleAVideo) require explicit `restorax download-models`.
- Silent stub fallbacks are removed — load failures raise `RestorerLoadError` with a clear install hint.
- A `restorax download-models` CLI handles batch weight fetching and status reporting.
- Real inference is validated end-to-end using standard academic benchmark assets.

---

## Architecture: Three Layers

### Layer 1 — Dependencies (`pyproject.toml`)

Four new optional groups isolate install cost by category:

```toml
[project.optional-dependencies]

sr = [
    "timm>=0.9.0",          # DDColor backbone, MambaIR
    "einops>=0.7.0",         # VRT, MambaIR, EvTexture tensor ops
    "mamba-ssm>=2.0.3",      # MambaIR selective state-space (CUDA required)
]

face = [
    "facexlib>=0.3.0",       # face detection/alignment for GFPGAN, CodeFormer
    "einops>=0.7.0",
]

diffusion = [
    "diffusers>=0.27.0",
    "transformers>=4.40.0",
    "accelerate>=0.30.0",
    "sentencepiece>=0.2.0",
]

extras = [
    "einops>=0.7.0",
    "kornia>=0.7.0",         # optical flow for GAVS, DeepFlowStab
    "imageio[ffmpeg]>=2.34.0",
]

all = ["restorax[sr,face,diffusion,extras,apm]"]
```

`mamba-ssm` requires a CUDA build environment. If it fails to import, `MambaIRRestorer.load()`
raises `RestorerLoadError("MambaIR requires mamba-ssm built with CUDA")` — no CPU fallback.

### Layer 2 — Arch Files (15 new `*_arch.py` files)

Each arch file lives beside its restorer module. Top of every file carries:

```python
# Vendored from <org/repo> (<SPDX license>). Adapted for RestoraX weight compatibility.
```

Imports are made self-contained — no relative imports into the source repo's internal structure.
Each file exports exactly one class: the `nn.Module` (or diffusion pipeline wrapper) that the
restorer already expects.

### Layer 3 — Restorer Wiring

Silent `try/except ImportError → stub` blocks are replaced with explicit `RestorerLoadError`.
Weight resolution follows the hybrid pattern (see Section 4). Audio restorers (Demucs, VoiceFixer,
RNNoise) keep their passthrough stubs — audio is genuinely optional and video output remains valid.

---

## Arch File Vendoring Plan

| Arch file | Source repo | License | Notes |
| --- | --- | --- | --- |
| `super_resolution/vrt_arch.py` | JingyunLiang/VRT (BasicSR) | Apache-2.0 | In BasicSR GitHub, not in pip release |
| `super_resolution/waifu2x_arch.py` | nagadomi/waifu2x | MIT | UpConvNet architecture |
| `super_resolution/mamba_ir_arch.py` | csguoh/MambaIR | Apache-2.0 | Requires `mamba-ssm` |
| `super_resolution/evtexture_arch.py` | researchmm/EvTexture | Apache-2.0 | Event-guided texture net |
| `super_resolution/flashvsr_arch.py` | official FlashVSR repo | MIT | Lightweight 3×3 conv + subpixel |
| `super_resolution/seedvr_arch.py` | ByteDance/SeedVR | Apache-2.0 | Wraps `diffusers` pipeline |
| `super_resolution/tdm_arch.py` | TDM official repo | MIT | Wraps `diffusers` pipeline |
| `super_resolution/upscale_a_video_arch.py` | sczhou/Upscale-A-Video | S-Lab License | Wraps `diffusers` pipeline |
| `colorization/ddcolor_arch.py` | piddnad/DDColor | Apache-2.0 | Uses `timm` backbone |
| `hdr/hdrtvdm_arch.py` | chxy95/HDRTVNet | MIT | HDR tone-mapping network |
| `face_restoration/dicface_arch.py` | YaNgZhAnG-NJU/DicFace | MIT | Dictionary-based face restoration |
| `face_restoration/codeformer_pp_arch.py` | CodeFormer++ repo | MIT | CodeFormer variant |
| `deinterlacing/deinterlace_arch.py` | tonycaisy/deinterlace-net | MIT | DeinterlaceNet |
| `stabilization/gavs_arch.py` | Annbless/GAVS | MIT | GAVS optical flow stabilization |
| `artifact_removal/propainter_arch.py` | sczhou/ProPainter | S-Lab License | Video inpainting pipeline |

---

## Weight Download Strategy (Hybrid)

### Lazy auto-download on first `load()` — models under 500 MB

Each restorer calls `_try_resolve_weight_path()` which checks `settings.model_dir/<name>/` then
downloads from HuggingFace Hub if the file is absent. Raises `RestorerLoadError` if HF is unreachable.

| Model | HF Repo | Weight file | ~Size |
| --- | --- | --- | --- |
| RealESRGAN | xinntao/Real-ESRGAN | RealESRGANx4plus.pth | 67 MB |
| BasicVSR++ | sczhou/BasicVSR-PlusPlus | basicvsr_pp_reds4.pth | 20 MB |
| VRT | JingyunLiang/VRT | VRT_videosr_bi_REDS_6frames.pth | 350 MB |
| EvTexture | researchmm/EvTexture | evtexture_x4.pth | ~80 MB |
| FlashVSR | FlashVSR/FlashVSR | flashvsr_x4.pth | ~15 MB |
| Waifu2x | nagadomi/waifu2x | waifu2x_art_noise1_scale2x.pth | ~5 MB |
| MambaIR | csguoh/MambaIR | mambair_sr_x4.pth | ~80 MB |
| GFPGAN | TencentARC/GFPGAN | GFPGANv1.4.pth | 330 MB |
| CodeFormer | sczhou/CodeFormer | codeformer.pth | 375 MB |
| CodeFormer++ | CodeFormerPP/weights | codeformer_pp.pth | ~380 MB |
| DicFace | YaNgZhAnG-NJU/DicFace | dicface.pth | ~200 MB |
| DDColor | piddnad/DDColor | ddcolor_modelscope.pth | ~850 MB† |
| HDRTVDM | chxy95/HDRTVNet | hdrtvdm.pth | ~50 MB |
| GAVS | Annbless/GAVS | gavs.pth | ~120 MB |
| DeinterlaceNet | tonycaisy/deinterlace-net | deinterlace.pth | ~30 MB |
| ProPainter | sczhou/ProPainter | ProPainter.pth + raft-things.pth | ~400 MB |
| RIFE | — | flownet.pkl | ✅ exists |
| Demucs | — | htdemucs.th | auto via `demucs` package |
| VoiceFixer | — | — | auto via `voicefixer` package |
| RNNoise | — | — | auto via `noisereduce` package |

†DDColor (~850 MB) is lazy but prints a size warning before download.

### CLI-required — diffusion models

`SeedVR` (~7 GB), `TDM` (~5 GB), `UpscaleAVideo` (~5 GB). If weights are absent, `load()` raises:

```text
RestorerLoadError: SeedVR weights not found. Download with:
  restorax download-models --model seedvr
```

---

## CLI: `restorax download-models`

New subcommand added to the existing `restorax` CLI entry point.

```bash
restorax download-models                          # show all models + status table
restorax download-models --all                    # download all non-diffusion models
restorax download-models --group diffusion        # download SeedVR, TDM, UpscaleAVideo
restorax download-models --group sr               # download all SR models
restorax download-models --model seedvr           # download one specific model
restorax download-models --model rife --force     # re-download even if present
```

**Output format:**

```text
Model           Group       Size     Status
──────────────────────────────────────────────
real_esrgan     sr          67 MB    ✓ ready
basicvsr_pp     sr          20 MB    ✗ missing
seedvr          diffusion   7.2 GB   ✗ missing (run --group diffusion)
rife            sr          12 MB    ✓ ready
...
```

Download shows a `tqdm` progress bar per file. Skips already-present weights unless `--force`.
Implemented in `restorax/cli/download_models.py`, registered under the `restorax` Click group.

---

## Stub Removal & Error Handling

### Before (silent stub)

```python
try:
    from restorax.restorers.super_resolution.evtexture_arch import EvTexture
except (ImportError, Exception) as exc:
    logger.info("EvTexture arch unavailable (%s) — bicubic stub", exc)
    self._model = _BicubicStub()
```

### After (explicit error)

```python
try:
    from restorax.restorers.super_resolution.evtexture_arch import EvTexture
except ImportError as exc:
    raise RestorerLoadError(
        "EvTexture requires restorax[sr]. Install with: pip install 'restorax[sr]'"
    ) from exc
```

### Three rules applied consistently

1. **Arch ImportError** → `RestorerLoadError` with pip install group hint
2. **Weights missing, lazy model** → `RestorerLoadError` with HF repo name
3. **Weights missing, diffusion model** → `RestorerLoadError` pointing to `restorax download-models --model <name>`

**Exception:** Audio restorers (Demucs, VoiceFixer, RNNoise) keep passthrough stubs. Audio
restoration is optional — video output is valid without it.

---

## Standard Test Assets

Assets fetched once into `tests/assets/` (gitignored). `conftest.py` session fixture
`download_test_assets()` checks each file and fetches from the canonical URL if absent.
Tests that need assets are auto-skipped if assets are missing (same pattern as weights).

| Asset | Source | License | Use |
| --- | --- | --- | --- |
| `vid4/calendar/`, `vid4/city/`, `vid4/foliage/`, `vid4/walk/` | Vid4 benchmark dataset | Academic | Video SR inference tests |
| `set5/` (baby, bird, butterfly, head, woman) | Set5 SR benchmark | Academic | Image restorer inference |
| `set14/` (14 PNG images) | Set14 SR benchmark | Academic | Benchmarking |
| `big_buck_bunny_360p_10s.mp4` | Blender Foundation | CC BY 3.0 | Full pipeline smoke test |
| `tears_of_steel_360p_10s.mp4` | Blender Foundation | CC BY-SA 3.0 | Video restorer diversity |
| `librispeech_dev_clean_sample.flac` | LibriSpeech dev-clean | CC BY 4.0 | Audio restorer inference |
| `vctk_p225_001.wav` | VCTK corpus | CC BY 4.0 | Audio restorer inference |

Assets are downloaded from their official URLs, not vendored into git.

---

## Testing Strategy

### Tier 1: Arch shape tests (`tests/unit/test_*_arch.py`)

One test file per arch. Instantiates each arch with random weights (no download), feeds a small
random tensor (e.g. `[1, 3, 64, 64]` for image archs, `[1, 4, 3, 64, 64]` for video archs),
asserts output shape and dtype. Fast, no network, runs in CI. ~28 tests.

### Tier 2: Stub removal tests (`tests/unit/test_restorer_error_handling.py`)

For each of the 14 previously-stubbed restorers, mocks the arch import to raise `ImportError`
and asserts `RestorerLoadError` is raised (not a silent fallback). ~14 tests.

### Tier 3: Real inference tests (`tests/integration/test_restorer_inference.py`)

Loads real weights, runs Set5 `butterfly.png` or 4 frames of `vid4/calendar/` through each
restorer on CPU or GPU (whichever `settings.device` resolves to). Asserts:

- Output shape matches expected upscale factor
- Output dtype is `uint8`
- No NaN or Inf values in output
- PSNR ≥ bicubic baseline on Set5 butterfly (models must beat a dumb baseline)

Tests are decorated with `@pytest.mark.requires_weights("modelname")` — a `conftest.py` plugin
reads `settings.model_dir` at collection time and auto-skips tests whose weights are absent.
Running locally after `restorax download-models --all` validates the full non-diffusion stack.

### Tier 4: Benchmark tests (`tests/benchmark/test_restorer_benchmark.py`)

Runs each restorer on `big_buck_bunny_360p_10s.mp4` (10 s, 360p). Records per-model:

- PSNR and SSIM vs. Set14 reference
- Wall-clock inference time (seconds per frame)
- Peak VRAM usage (MB)

Results written to `tests/benchmark/results/YYYY-MM-DD.json` for trend tracking.
Excluded from default `pytest` run — invoked with `pytest -m benchmark`.

### Tier 5: CLI tests (`tests/integration/test_download_models.py`)

- `restorax download-models` (no args) prints status table without error
- `--model rife` skips download when `models/rife/flownet.pkl` exists
- `--model basicvsr_pp` triggers download when weights absent (network mocked via `pytest-mock`)
- `--group diffusion` without `--force` prompts size warning

---

## File Map

| Action | File | Purpose |
| --- | --- | --- |
| Modify | `pyproject.toml` | Add `[sr]`, `[face]`, `[diffusion]`, `[extras]`, `[all]` groups |
| Create | `restorax/cli/download_models.py` | `restorax download-models` subcommand |
| Modify | `restorax/cli/__init__.py` | Register `download_models` command |
| Create | `restorax/restorers/super_resolution/vrt_arch.py` | VRT architecture |
| Create | `restorax/restorers/super_resolution/waifu2x_arch.py` | Waifu2x UpConvNet |
| Create | `restorax/restorers/super_resolution/mamba_ir_arch.py` | MambaIR SSM |
| Create | `restorax/restorers/super_resolution/evtexture_arch.py` | EvTexture |
| Create | `restorax/restorers/super_resolution/flashvsr_arch.py` | FlashVSR |
| Create | `restorax/restorers/super_resolution/seedvr_arch.py` | SeedVR diffusion wrapper |
| Create | `restorax/restorers/super_resolution/tdm_arch.py` | TDM diffusion wrapper |
| Create | `restorax/restorers/super_resolution/upscale_a_video_arch.py` | UpscaleAVideo wrapper |
| Create | `restorax/restorers/colorization/ddcolor_arch.py` | DDColor U-Net |
| Create | `restorax/restorers/hdr/hdrtvdm_arch.py` | HDRTVNet |
| Create | `restorax/restorers/face_restoration/dicface_arch.py` | DicFace net |
| Create | `restorax/restorers/face_restoration/codeformer_pp_arch.py` | CodeFormer++ |
| Create | `restorax/restorers/deinterlacing/deinterlace_arch.py` | DeinterlaceNet |
| Create | `restorax/restorers/stabilization/gavs_arch.py` | GAVS pipeline |
| Create | `restorax/restorers/artifact_removal/propainter_arch.py` | ProPainter pipeline |
| Modify | `restorax/restorers/super_resolution/real_esrgan.py` | Remove stub, wire weights |
| Modify | `restorax/restorers/super_resolution/basicvsr_pp.py` | Remove stub, wire weights |
| Modify | `restorax/restorers/super_resolution/vrt.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/waifu2x.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/mamba_ir.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/evtexture.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/flashvsr.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/seedvr.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/tdm.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/super_resolution/upscale_a_video.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/colorization/ddcolor.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/hdr/hdrtvdm.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/face_restoration/codeformer.py` | Remove stub, wire weights |
| Modify | `restorax/restorers/face_restoration/codeformer_pp.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/face_restoration/dicface.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/face_restoration/gfpgan.py` | Remove stub, wire weights |
| Modify | `restorax/restorers/deinterlacing/ai_deinterlace.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/stabilization/gavs.py` | Remove stub, wire arch + weights |
| Modify | `restorax/restorers/artifact_removal/scratch_removal.py` | Remove stub, wire arch + weights |
| Create | `tests/conftest_assets.py` | `download_test_assets()` session fixture |
| Modify | `tests/conftest.py` | Register `requires_weights` mark + asset fixture |
| Create | `tests/unit/test_vrt_arch.py` | VRT arch shape test |
| Create | `tests/unit/test_waifu2x_arch.py` | Waifu2x arch shape test |
| Create | `tests/unit/test_mamba_ir_arch.py` | MambaIR arch shape test |
| Create | `tests/unit/test_evtexture_arch.py` | EvTexture arch shape test |
| Create | `tests/unit/test_flashvsr_arch.py` | FlashVSR arch shape test |
| Create | `tests/unit/test_ddcolor_arch.py` | DDColor arch shape test |
| Create | `tests/unit/test_hdrtvdm_arch.py` | HDRTVDM arch shape test |
| Create | `tests/unit/test_dicface_arch.py` | DicFace arch shape test |
| Create | `tests/unit/test_codeformer_pp_arch.py` | CodeFormer++ arch shape test |
| Create | `tests/unit/test_deinterlace_arch.py` | DeinterlaceNet arch shape test |
| Create | `tests/unit/test_gavs_arch.py` | GAVS arch shape test |
| Create | `tests/unit/test_propainter_arch.py` | ProPainter arch shape test |
| Create | `tests/unit/test_restorer_error_handling.py` | Stub removal verification |
| Create | `tests/integration/test_restorer_inference.py` | Real inference on standard assets |
| Create | `tests/integration/test_download_models.py` | CLI tests |
| Create | `tests/benchmark/test_restorer_benchmark.py` | PSNR/SSIM/timing benchmark |
| Create | `tests/benchmark/results/.gitkeep` | Results directory |

---

## Out of Scope

- Model fine-tuning or training
- Custom preset YAML files for new models (handled in Track 3)
- Frontend model browser UI (Track 4)
- Doc updates (Track 5)
