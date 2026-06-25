from restorax.restorers.frame_interpolation.rife import RIFERestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_RIFE": make_restorer_node(RIFERestorer, "Frame Interpolation")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_RIFE": "RestoraX RIFE"}
