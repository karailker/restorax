from restorax.restorers.enhancement.dlss5_visual_enhancer import DLSS5VisualEnhancerRestorer

from ._base import make_restorer_node

_CATEGORY = "Enhancement"
NODE_CLASS_MAPPINGS = {
    "RestoraX_DLSS5VisualEnhancer": make_restorer_node(DLSS5VisualEnhancerRestorer, _CATEGORY),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RestoraX_DLSS5VisualEnhancer": "RestoraX DLSS 5 Neural Rendering (external)",
}
