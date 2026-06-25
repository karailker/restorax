from restorax.restorers.hdr.hdrtvdm import HDRTVDMRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {"RestoraX_HDRTVDM": make_restorer_node(HDRTVDMRestorer, "HDR Conversion")}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_HDRTVDM": "RestoraX HDRTVDM"}
