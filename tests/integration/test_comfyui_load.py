"""
Verifies the comfyui_nodes pack loads cleanly inside a real ComfyUI checkout.

Setup (one-time, manual — not run by CI):
    pip install comfy-cli
    comfy --skip-prompt install --fast-deps  # clones ComfyUI to ./ComfyUI by default
    export COMFYUI_PATH=$(pwd)/ComfyUI
    ln -s $(pwd)/comfyui_nodes $COMFYUI_PATH/custom_nodes/restorax
    pytest tests/integration/test_comfyui_load.py -v
"""
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.requires_comfyui


def test_comfyui_nodes_pack_importable_from_custom_nodes_dir():
    comfyui_path = os.environ["COMFYUI_PATH"]
    custom_nodes_link = os.path.join(comfyui_path, "custom_nodes", "restorax")
    assert os.path.islink(custom_nodes_link) or os.path.isdir(custom_nodes_link), (
        f"Expected comfyui_nodes/ symlinked into {custom_nodes_link} — see module docstring for setup."
    )
    result = subprocess.run(
        [sys.executable, "-c", "import comfyui_nodes; print(len(comfyui_nodes.NODE_CLASS_MAPPINGS))"],
        cwd=custom_nodes_link,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "22"
