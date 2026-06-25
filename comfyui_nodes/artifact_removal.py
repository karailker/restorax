from restorax.restorers.artifact_removal.scratch_removal import ScratchRemovalRestorer

from ._base import make_restorer_node

NODE_CLASS_MAPPINGS = {
    "RestoraX_ScratchRemoval": make_restorer_node(ScratchRemovalRestorer, "Artifact Removal"),
}
NODE_DISPLAY_NAME_MAPPINGS = {"RestoraX_ScratchRemoval": "RestoraX Scratch Removal"}
