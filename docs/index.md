# RestoraX

**Modern AI video restoration platform for old films, series, and archival footage.**

RestoraX combines 25 state-of-the-art deep learning models into a single modular pipeline:
upscale, colorize, restore faces, remove scratches, deinterlace, stabilize, boost frame rate,
and convert to HDR — all orchestrated by a REST API, Pipeline DAG engine, visual node builder,
ComfyUI node pack, and CLI.

## Quick links

- [Installation](guides/installation.md)
- [Quick Start](guides/quickstart.md)
- [Pipeline Presets](presets/overview.md)
- [Writing a Plugin](guides/plugins.md)

## Feature matrix

| Category | Models | Scale |
|---|---|---|
| **Super-Resolution** (10) | Real-ESRGAN, BasicVSR++, VRT, MambaIR, Upscale-A-Video, TDM, SeedVR, EVTexture, FlashVSR, Waifu2x | 2×–8× |
| **Colorization** (1) | DDColor | 1× |
| **Face Restoration** (4) | CodeFormer, CodeFormer++, GFPGAN, DICFace | 1× |
| **Frame Interpolation** (1) | RIFE v4.22 | 2× FPS |
| **Scratch & Dust** (1) | ProPainter | 1× |
| **Deinterlacing** (2) | AI Deinterlacer + YADIF | 1× |
| **Stabilization** (2) | GaVS + Deep Flow | 1× |
| **SDR → HDR** (1) | HDRTVDM (CVPR 2023) | 1× |
| **Audio Restoration** (3) | Demucs, VoiceFixer, RNNoise | — |

## Architecture highlights

- **Pipeline DAG engine** — typed ports, parallel branches, merge strategies, retry policies
- **React Flow visual builder** — drag-and-drop node canvas at `/builder`
- **ComfyUI node pack** — custom nodes for every restorer, maintained in the separate [restorax-comfyui](https://github.com/karailker/restorax-comfyui) repository (linked here as the `comfyui_nodes/` git submodule)
- **REST API + WebSocket** — submit jobs, stream progress, download results
- **CLI** — single-command restoration without running the server
