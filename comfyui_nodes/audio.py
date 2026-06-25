from restorax.restorers.audio.demucs import DemucsRestorer
from restorax.restorers.audio.rnnoise import RNNoiseRestorer
from restorax.restorers.audio.voicefixer import VoiceFixerRestorer

from ._base import make_audio_restorer_node

_CATEGORY = "Audio"
_RESTORERS = [
    ("Demucs", DemucsRestorer, "RestoraX Demucs"),
    ("VoiceFixer", VoiceFixerRestorer, "RestoraX VoiceFixer"),
    ("RNNoise", RNNoiseRestorer, "RestoraX RNNoise"),
]

NODE_CLASS_MAPPINGS = {
    f"RestoraX_{key}": make_audio_restorer_node(cls, _CATEGORY) for key, cls, _ in _RESTORERS
}
NODE_DISPLAY_NAME_MAPPINGS = {f"RestoraX_{key}": label for key, _, label in _RESTORERS}
