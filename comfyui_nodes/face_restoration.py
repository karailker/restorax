from restorax.restorers.face_restoration.codeformer import CodeFormerRestorer
from restorax.restorers.face_restoration.codeformer_pp import CodeFormerPlusPlusRestorer
from restorax.restorers.face_restoration.dicface import DicFaceRestorer
from restorax.restorers.face_restoration.gfpgan import GFPGANRestorer

from ._base import make_restorer_node

_CATEGORY = "Face Restoration"
_RESTORERS = [
    ("CodeFormer", CodeFormerRestorer, "RestoraX CodeFormer"),
    ("CodeFormerPP", CodeFormerPlusPlusRestorer, "RestoraX CodeFormer++"),
    ("GFPGAN", GFPGANRestorer, "RestoraX GFPGAN"),
    ("DicFace", DicFaceRestorer, "RestoraX DicFace"),
]

NODE_CLASS_MAPPINGS = {
    f"RestoraX_{key}": make_restorer_node(cls, _CATEGORY) for key, cls, _ in _RESTORERS
}
NODE_DISPLAY_NAME_MAPPINGS = {f"RestoraX_{key}": label for key, _, label in _RESTORERS}
