from restorax.restorers.deinterlacing.ai_deinterlace import AIDeinterlaceRestorer
from restorax.restorers.deinterlacing.yadif_deinterlace import YadifDeinterlaceRestorer

from ._base import make_restorer_node

_CATEGORY = "Deinterlacing"
NODE_CLASS_MAPPINGS = {
    "RestoraX_AIDeinterlace": make_restorer_node(AIDeinterlaceRestorer, _CATEGORY),
    "RestoraX_Yadif": make_restorer_node(YadifDeinterlaceRestorer, _CATEGORY),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RestoraX_AIDeinterlace": "RestoraX AI Deinterlace",
    "RestoraX_Yadif": "RestoraX YADIF",
}
