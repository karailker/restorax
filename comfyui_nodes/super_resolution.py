from restorax.restorers.super_resolution.basicvsr_pp import BasicVSRPlusPlusRestorer
from restorax.restorers.super_resolution.evtexture import EvTextureRestorer
from restorax.restorers.super_resolution.flashvsr import FlashVSRRestorer
from restorax.restorers.super_resolution.mamba_ir import MambaIRRestorer
from restorax.restorers.super_resolution.real_esrgan import RealESRGANx4Restorer
from restorax.restorers.super_resolution.seedvr import SeedVRRestorer
from restorax.restorers.super_resolution.tdm import TDMRestorer
from restorax.restorers.super_resolution.upscale_a_video import UpscaleAVideoRestorer
from restorax.restorers.super_resolution.vrt import VRTRestorer
from restorax.restorers.super_resolution.waifu2x import Waifu2xRestorer

from ._base import make_restorer_node

_CATEGORY = "Super Resolution"
_RESTORERS = [
    ("RealESRGAN", RealESRGANx4Restorer, "RestoraX Real-ESRGAN x4"),
    ("BasicVSRPP", BasicVSRPlusPlusRestorer, "RestoraX BasicVSR++"),
    ("UpscaleAVideo", UpscaleAVideoRestorer, "RestoraX Upscale-A-Video"),
    ("VRT", VRTRestorer, "RestoraX VRT"),
    ("MambaIR", MambaIRRestorer, "RestoraX MambaIR"),
    ("TDM", TDMRestorer, "RestoraX TDM"),
    ("SeedVR", SeedVRRestorer, "RestoraX SeedVR"),
    ("Waifu2x", Waifu2xRestorer, "RestoraX Waifu2x"),
    ("FlashVSR", FlashVSRRestorer, "RestoraX FlashVSR"),
    ("EvTexture", EvTextureRestorer, "RestoraX EvTexture"),
]

NODE_CLASS_MAPPINGS = {
    f"RestoraX_{key}": make_restorer_node(cls, _CATEGORY) for key, cls, _ in _RESTORERS
}
NODE_DISPLAY_NAME_MAPPINGS = {f"RestoraX_{key}": label for key, _, label in _RESTORERS}
