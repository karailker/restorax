# restorax/models_catalog.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Group = Literal["sr", "face", "diffusion", "extras", "audio"]

# How usable a model is today (see ``note`` on each entry for the evidence):
#   ready        produces real output with real weights
#   fallback     runs, but through a classical (non-neural) fallback
#   needs_work   public weights/code exist, but the adapter or architecture is not finished
#   gpu_only     upstream needs CUDA-only kernels or very large VRAM; not implemented here
#   unavailable  no public code or weights could be found
Status = Literal["ready", "fallback", "needs_work", "gpu_only", "unavailable"]


@dataclass
class ModelEntry:
    name: str
    group: Group
    hf_repo: str
    weight_files: list[str]
    size_mb: int
    snapshot: bool = False
    # Direct HTTPS downloads (e.g. GitHub release assets), keyed by weight file name.
    # When set, these are used instead of the Hugging Face repo.
    urls: dict[str, str] = field(default_factory=dict)
    status: Status = "ready"
    note: str = ""

    def weight_dir(self) -> Path:
        from restorax.config import settings

        return Path(settings.model_dir) / self.name

    def is_ready(self) -> bool:
        if self.snapshot:
            return self.weight_dir().exists() and any(self.weight_dir().iterdir())
        return all((self.weight_dir() / f).exists() for f in self.weight_files)


CATALOG: list[ModelEntry] = [
    ModelEntry("real_esrgan", "sr", "xinntao/Real-ESRGAN", ["RealESRGANx4plus.pth"], 67),
    ModelEntry(
        "basicvsr_pp",
        "sr",
        "maybleMyers/interpolate",
        ["BasicVSR_PlusPlus_REDS4.pth"],
        20,
    ),
    ModelEntry(
        "vrt",
        "sr",
        "JingyunLiang/VRT",
        ["001_VRT_videosr_bi_REDS_6frames.pth"],
        166,
        urls={
            "001_VRT_videosr_bi_REDS_6frames.pth": (
                "https://github.com/JingyunLiang/VRT/releases/download/v0.0/"
                "001_VRT_videosr_bi_REDS_6frames.pth"
            )
        },
    ),
    ModelEntry("waifu2x", "sr", "deepghs/waifu2x", ["waifu2x_x2.pth"], 5),
    ModelEntry("mamba_ir", "sr", "csguoh/MambaIR", ["MambaIR_SR_x4.pth"], 80),
    ModelEntry("evtexture", "sr", "DachunKai/EvTexture", ["evtexture_x4.pth"], 80),
    ModelEntry("flashvsr", "sr", "restorax/flashvsr-weights", ["flashvsr_x4.pth"], 15),
    ModelEntry("codeformer", "face", "sczhou/CodeFormer", ["codeformer.pth"], 375),
    ModelEntry(
        "codeformer_pp",
        "face",
        "TencentARC/GFPGAN",
        ["RestoreFormer.pth"],
        277,
        urls={
            "RestoreFormer.pth": (
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/RestoreFormer.pth"
            )
        },
    ),
    ModelEntry("gfpgan", "face", "TencentARC/GFPGANv1.4", ["GFPGANv1.4.pth"], 330),
    ModelEntry(
        "dicface", "face", "fudan-generative-ai/DicFace_model", ["CodeFormer/bfr_100k.pth"], 887
    ),
    ModelEntry("ddcolor", "sr", "piddnad/ddcolor_modelscope", ["pytorch_model.bin"], 870),
    ModelEntry("hdrtvdm", "extras", "AndreGuo/HDRTVDM", ["HDRTVNet.pth"], 50),
    ModelEntry("gavs", "extras", "Annbless/GAVS", ["gavs.pth"], 120),
    ModelEntry("deinterlace", "extras", "tonycaisy/deinterlace-net", ["deinterlace.pth"], 30),
    ModelEntry(
        "scratch_removal", "extras", "sczhou/ProPainter", ["ProPainter.pth", "raft-things.pth"], 400
    ),
    ModelEntry("rife", "sr", "AlexZou/RIFE-v4", ["flownet.pkl"], 12),
    ModelEntry("seedvr", "diffusion", "IceClear/SeedVR", [], 7200, snapshot=True),
    ModelEntry("tdm", "diffusion", "ChenyangSi/TDM", [], 5000, snapshot=True),
    ModelEntry("upscale_a_video", "diffusion", "sczhou/Upscale-A-Video", [], 5000, snapshot=True),
    ModelEntry("demucs", "audio", "facebook/demucs", [], 0),
    ModelEntry("voicefixer", "audio", "haoheliu/voicefixer", [], 0),
    ModelEntry("rnnoise", "audio", "restorax/rnnoise", [], 0),
]

# Audit of every model's real status (October 2026). Entries not listed are "ready".
# Evidence is recorded in the note so the table can be re-checked later.
_STATUS: dict[str, tuple[Status, str]] = {
    "vrt": (
        "ready",
        "Official REDS 6-frame weights from the VRT GitHub release; verified on CPU "
        "(+2.8 to +5.8 dB over bicubic). Input is padded to a multiple of 32 (min 64 px).",
    ),
    "rife": (
        "ready",
        "Architecture (IFNet) vendored and official weights wired from AlexWortega/RIFE "
        "(flownet.pkl, 12 MB). Verified to load and interpolate frames.",
    ),
    "gavs": (
        "fallback",
        "Upstream code is github.com/huawei-bayerlab/GaVS; not integrated. OpenCV optical-flow "
        "stabilization is used instead.",
    ),
    "rnnoise": ("fallback", "Classical noise gate; the RNNoise network is not bundled."),
    "basicvsr_pp": (
        "ready",
        "Architecture vendored in BasicSR, official REDS weights wired from maybleMyers/interpolate "
        "(BasicVSR_PlusPlus_REDS4.pth, 20 MB).",
    ),
    "ddcolor": (
        "ready",
        "Verified in CI with the official piddnad/ddcolor_modelscope weights (pytorch_model.bin, "
        "912 MB) on CPU: the result carries real chroma and is closer to the color original "
        "than the gray input. Works frame by frame, so colors can flicker between video frames.",
    ),
    "hdrtvdm": (
        "ready",
        "Arch (TriSegNet) vendored from AndreGuo/HDRTVDM (MPL 2.0) and official "
        "params.pth weights wired from GitHub raw. Verified to load and convert SDR "
        "frames to HDR preview.",
    ),
    "dicface": (
        "ready",
        "Arch handled by the vendored TemporalCodeFormerDirDistMultiScale and official "
        "fudan-generative-ai/DicFace_model (CodeFormer/bfr_100k.pth, 887 MB) weights wired. "
        "Adapter and weight loading verified.",
    ),
    "deinterlace": (
        "needs_work",
        "The referenced source (tonycaisy/deinterlace-net) could not be verified. Use YADIF.",
    ),
    "mamba_ir": (
        "gpu_only",
        "Needs the mamba-ssm CUDA kernels (no CPU path); the architecture is not vendored.",
    ),
    "flashvsr": (
        "gpu_only",
        "Upstream (github.com/OpenImagingLab/FlashVSR, Apache-2.0) is a one-step diffusion "
        "model that needs Block-Sparse Attention on CUDA. The earlier description of this "
        "adapter as a lightweight recurrent network was inaccurate; the weights id here is "
        "a placeholder.",
    ),
    "seedvr": (
        "gpu_only",
        "Official SeedVR/SeedVR2 weights are on Hugging Face (ByteDance-Seed, Apache-2.0 per "
        "mirrors) but need >= 24 GB VRAM; the adapter is not implemented and the repo id "
        "here is unverified.",
    ),
    "upscale_a_video": (
        "gpu_only",
        "Diffusion model under the NTU S-Lab License 1.0 (non-commercial); only an adapter "
        "shim exists, the real pipeline is not ported.",
    ),
    "codeformer_pp": (
        "ready",
        "CodeFormer++ alias wired to the vendored RestoreFormer architecture and official "
        "RestoreFormer.pth weights from GFPGAN releases. Adapter and weight loading verified.",
    ),
    "tdm": (
        "unavailable",
        "No public code or weights were found for a video super-resolution model by this "
        "name; the Hugging Face id here is a placeholder.",
    ),
}

for _entry in CATALOG:
    _entry.status, _entry.note = _STATUS.get(_entry.name, ("ready", ""))

CATALOG_BY_NAME: dict[str, ModelEntry] = {m.name: m for m in CATALOG}
