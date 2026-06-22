"""GET /models — list available restorers and their capabilities."""
from __future__ import annotations

from fastapi import APIRouter

from dataclasses import asdict

from restorax.api.schemas.model import ModelListResponse, ParamSpecSchema, RestorerInfo
from restorax.audio.restorer import AudioRestorerCapabilities
from restorax.core.restorer import RestorerCapabilities
from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer
from restorax.restorers.face_restoration.dicface import DicFaceRestorer
from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer
from restorax.restorers.colorization.ddcolor import DDColorRestorer
from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer
from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer
from restorax.restorers.frame_interpolation.rife import RIFERestorer
from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer
from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
from restorax.restorers.stabilization.gavs import GaVSRestorer
from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer
from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from restorax.restorers.super_resolution.tdm import TDMRestorer
from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
from restorax.restorers.super_resolution.vrt import VRTRestorer
from restorax.restorers.audio.demucs import DemucsRestorer
from restorax.restorers.audio.voicefixer import VoiceFixerRestorer
from restorax.restorers.audio.rnnoise import RNNoiseRestorer

router = APIRouter(prefix="/models", tags=["models"])

_RESTORER_CLASSES = [
    RealESRGANx4Restorer, BasicVSRPlusPlusRestorer, UpscaleAVideoRestorer,
    VRTRestorer, MambaIRRestorer, TDMRestorer, SeedVRRestorer,
    Waifu2xRestorer, FlashVSRRestorer, EvTextureRestorer,
    CodeFormerRestorer, CodeFormerPlusPlusRestorer, GFPGANRestorer, DicFaceRestorer,
    DDColorRestorer, RIFERestorer,
    ScratchRemovalRestorer, HDRTVDMRestorer, VideoStabilizationRestorer,
    GaVSRestorer, AIDeinterlaceRestorer, YadifDeinterlaceRestorer,
    DemucsRestorer, VoiceFixerRestorer, RNNoiseRestorer,
]


@router.get("", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    restorers = []
    for cls in _RESTORER_CLASSES:
        instance = object.__new__(cls)  # FRAGILE: assumes capabilities is a pure property with no instance state
        caps = cls.capabilities.fget(instance)  # type: ignore[attr-defined]
        param_schema = [ParamSpecSchema(**asdict(spec)) for spec in cls.PARAM_SCHEMA]

        if isinstance(caps, RestorerCapabilities):
            # Video restorer (color space aware)
            restorers.append(
                RestorerInfo(
                    name=cls.name.fget(instance),  # type: ignore[attr-defined]
                    kind="video",
                    category=caps.category.value,
                    input_color_space=caps.input_color_space,
                    output_color_space=caps.output_color_space,
                    requires_temporal=caps.requires_temporal,
                    min_vram_gb=caps.min_vram_gb,
                    scale_factor=caps.scale_factor,
                    tags=caps.tags,
                    loaded=False,  # Phase 3: wire into live registry
                    param_schema=param_schema,
                )
            )
        elif isinstance(caps, AudioRestorerCapabilities):
            # Audio restorer (sample rate aware)
            restorers.append(
                RestorerInfo(
                    name=cls.name.fget(instance),  # type: ignore[attr-defined]
                    kind="audio",
                    category=caps.category.value,
                    min_ram_gb=caps.min_ram_gb,
                    supports_stereo=caps.supports_stereo,
                    sample_rates=caps.sample_rates,
                    tags=caps.tags,
                    loaded=False,  # Phase 3: wire into live registry
                    param_schema=param_schema,
                )
            )
        else:
            raise TypeError(f"Unrecognised capabilities type for {cls!r}: {type(caps)}")
    return ModelListResponse(restorers=restorers)
