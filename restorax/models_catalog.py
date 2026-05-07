# restorax/models_catalog.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Group = Literal["sr", "face", "diffusion", "extras", "audio"]


@dataclass
class ModelEntry:
    name: str
    group: Group
    hf_repo: str
    weight_files: list[str]
    size_mb: int
    snapshot: bool = False

    def weight_dir(self) -> Path:
        from restorax.config import settings
        return Path(settings.model_dir) / self.name

    def is_ready(self) -> bool:
        if self.snapshot:
            return self.weight_dir().exists() and any(self.weight_dir().iterdir())
        return all((self.weight_dir() / f).exists() for f in self.weight_files)


CATALOG: list[ModelEntry] = [
    ModelEntry("real_esrgan", "sr", "xinntao/Real-ESRGAN", ["RealESRGANx4plus.pth"], 67),
    ModelEntry("basicvsr_pp", "sr", "sczhou/BasicVSR-PlusPlus", ["BasicVSR_PlusPlus_REDS4.pth"], 20),
    ModelEntry("vrt", "sr", "JingyunLiang/VRT", ["VRT_videosr_bi_Vimeo_7frames.pth"], 350),
    ModelEntry("waifu2x", "sr", "deepghs/waifu2x", ["waifu2x_x2.pth"], 5),
    ModelEntry("mamba_ir", "sr", "csguoh/MambaIR", ["MambaIR_SR_x4.pth"], 80),
    ModelEntry("evtexture", "sr", "DachunKai/EvTexture", ["evtexture_x4.pth"], 80),
    ModelEntry("flashvsr", "sr", "restorax/flashvsr-weights", ["flashvsr_x4.pth"], 15),
    ModelEntry("codeformer", "face", "sczhou/CodeFormer", ["codeformer.pth"], 375),
    ModelEntry("codeformer_pp", "face", "sczhou/CodeFormerPlusPlus", ["codeformer_pp.pth"], 380),
    ModelEntry("gfpgan", "face", "TencentARC/GFPGANv1.4", ["GFPGANv1.4.pth"], 330),
    ModelEntry("dicface", "face", "YaNgZhAnG-V5/DicFace", ["dicface.pth"], 200),
    ModelEntry("ddcolor", "sr", "piddnad/ddcolor_models", ["ddcolor_artistic.pth"], 850),
    ModelEntry("hdrtvdm", "extras", "AndreGuo/HDRTVDM", ["HDRTVNet.pth"], 50),
    ModelEntry("gavs", "extras", "Annbless/GAVS", ["gavs.pth"], 120),
    ModelEntry("deinterlace", "extras", "tonycaisy/deinterlace-net", ["deinterlace.pth"], 30),
    ModelEntry("scratch_removal", "extras", "sczhou/ProPainter", ["ProPainter.pth", "raft-things.pth"], 400),
    ModelEntry("rife", "sr", "AlexZou/RIFE-v4", ["flownet.pkl"], 12),
    ModelEntry("seedvr", "diffusion", "IceClear/SeedVR", [], 7200, snapshot=True),
    ModelEntry("tdm", "diffusion", "ChenyangSi/TDM", [], 5000, snapshot=True),
    ModelEntry("upscale_a_video", "diffusion", "sczhou/Upscale-A-Video", [], 5000, snapshot=True),
    ModelEntry("demucs", "audio", "facebook/demucs", [], 0),
    ModelEntry("voicefixer", "audio", "haoheliu/voicefixer", [], 0),
    ModelEntry("rnnoise", "audio", "restorax/rnnoise", [], 0),
]

CATALOG_BY_NAME: dict[str, ModelEntry] = {m.name: m for m in CATALOG}
