# Vendored reference for https://github.com/ByteDance-Seed/SeedVR (Apache License 2.0)
# Canonical org repo — hosts SeedVR (CVPR2025 Highlight) and SeedVR2 (ICLR2026).
# Weights (Apache-2.0, real, downloadable): huggingface.co/ByteDance-Seed/SeedVR-7B,
# huggingface.co/ByteDance-Seed/SeedVR2-3B.
#
# ponytail: This is a minimal ADAPTER shim, not a faithful port — and it CANNOT
# become a diffusers-style `from_pretrained` pipeline no matter how much is ported,
# because upstream SeedVR isn't built on diffusers at all. It ships as a raw
# checkpoint (`ema_vae.pth` + `seedvr_ema_7b.pth`, no `model_index.json`) loaded by
# custom inference scripts (`projects/inference_seedvr2_3b.py` etc.) that wire up:
#   - a causal video VAE (custom, not diffusers AutoencoderKL)
#   - a Swin-MMDiT diffusion transformer (shifted-window attention, 2.48B+ params)
#   - custom multi-GPU FSDP sharding for inference
#   - apex / flash-attn compiled CUDA extensions (prebuilt wheels ship on HF, not
#     installable via plain pip)
# None of that is vendored here — porting it is out of scope for a single-session
# change (unlike Upscale-A-Video, which at least follows a diffusers-shaped
# pipeline). `from_pretrained` raises ImportError — caught by
# `seedvr.py::_build_pipeline` and converted to a clean RestorerLoadError —
# rather than silently faking a working diffusion pipeline.
from __future__ import annotations

import diffusers  # noqa: F401 — gate: real upstream needs apex/flash-attn/custom VAE+DiT, not diffusers, but this keeps the same "missing optional extra" failure path as other vendored diffusion restorers
import torch


class SeedVRPipeline:
    """Adapter matching the call surface `SeedVRRestorer` expects.

    Real upstream has no equivalent single class: inference is driven by a
    standalone script that loads `seedvr_ema_7b.pth` / `ema_vae.pth` directly
    into hand-built VAE + Swin-MMDiT modules, not a `from_pretrained` pipeline.
    """

    def __init__(self, vae: object | None = None, dit: object | None = None) -> None:
        self.vae = vae
        self.dit = dit
        self.device = torch.device("cpu")

    def to(self, device: torch.device) -> "SeedVRPipeline":
        self.device = device
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str) -> "SeedVRPipeline":
        raise ImportError(
            "SeedVR's real weights (causal video VAE + Swin-MMDiT transformer, "
            "apex/flash-attn CUDA kernels) are not vendored in restorax — only "
            "this adapter shim is. Real weight loading is unimplemented."
        )

    def __call__(self, image, num_inference_steps: int = 25, guidance_scale: float = 7.5):
        raise NotImplementedError(
            "SeedVRPipeline inference requires the un-vendored upstream causal "
            "video VAE and Swin-MMDiT diffusion transformer."
        )
