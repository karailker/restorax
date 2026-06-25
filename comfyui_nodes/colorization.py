from restorax.restorers.colorization.ddcolor import DDColorRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_DDColor": make_restorer_node(DDColorRestorer, "Colorization")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_DDColor": "RestoraX DDColor"}
