from restorax.restorers.stabilization.deep_flow_stab import VideoStabilizationRestorer
from restorax.restorers.stabilization.gavs import GaVSRestorer

from ._base import make_restorer_node

_CATEGORY = "Stabilization"
NODE_CLASS_MAPPINGS = {
    "RestoraX_DeepFlowStab": make_restorer_node(VideoStabilizationRestorer, _CATEGORY),
    "RestoraX_GaVS": make_restorer_node(GaVSRestorer, _CATEGORY),
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RestoraX_DeepFlowStab": "RestoraX Optical-Flow Stabilization",
    "RestoraX_GaVS": "RestoraX GaVS",
}
