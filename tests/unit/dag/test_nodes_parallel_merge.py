from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import numpy as np

from restorax.dag.nodes.merge import MergeNode
from restorax.dag.nodes.parallel import BranchConfig


def test_merge_node_blend_averages_frames():
    frame_a = np.full((4, 4, 3), 100, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 200, dtype=np.uint8)
    # branch_outputs: 2 branches, 1 chunk each, 1 frame each
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="blend")
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    merged_frame = result.outputs["chunks"][0][0]
    assert merged_frame.dtype == np.uint8
    np.testing.assert_allclose(merged_frame, 150, atol=1)


def test_merge_node_select_picks_correct_branch():
    frame_a = np.full((4, 4, 3), 10, dtype=np.uint8)
    frame_b = np.full((4, 4, 3), 99, dtype=np.uint8)
    branch_outputs = [[[frame_a]], [[frame_b]]]
    node = MergeNode(id="m1", name="Merge", strategy="select", select_index=1)
    result = asyncio.run(node.execute(MagicMock(), {"branch_outputs": branch_outputs}))
    np.testing.assert_array_equal(result.outputs["chunks"][0][0], frame_b)


def test_branch_config_roundtrip():
    bc = BranchConfig(name="branch_a", restorer_steps=[("real_esrgan", {"scale": 4})])
    restored = BranchConfig.from_dict(bc.to_dict())
    assert restored.name == "branch_a"
    assert restored.restorer_steps[0][0] == "real_esrgan"


def _parallel_ctx(tmp_path):
    """ExecutionContext whose registry returns an identity restorer per branch."""
    from restorax.core.restorer import (
        RestorerCapabilities,
        RestorerCategory,
        RestorerParams,
    )

    class _Identity:
        @property
        def capabilities(self):
            return RestorerCapabilities(
                category=RestorerCategory.SUPER_RESOLUTION,
                input_color_space="rgb",
                output_color_space="rgb",
            )

        def process_frame(self, frame, params):  # noqa: ANN001, ANN201
            return frame

    ctx = MagicMock()
    ctx.work_dir = tmp_path
    ctx.device = MagicMock()
    ctx.registry.get.return_value = _Identity()
    ctx.progress_emitter = MagicMock()
    _ = RestorerParams  # imported for parity with the node's params construction
    return ctx


def test_parallel_node_records_branch_metadata_and_previews(tmp_path):
    from restorax.dag.nodes.parallel import ParallelNode

    frame = np.full((4, 4, 3), 42, dtype=np.uint8)
    node = ParallelNode(
        id="par",
        name="Parallel",
        branches=[
            BranchConfig(name="sharp", restorer_steps=[("a", {})]),
            BranchConfig(name="soft", restorer_steps=[("b", {})]),
        ],
    )
    ctx = _parallel_ctx(tmp_path)
    result = asyncio.run(node.execute(ctx, {"chunks": [[frame]], "meta": {"fps": 24}}))

    branches = result.metadata["branches"]
    assert [b["name"] for b in branches] == ["sharp", "soft"]
    assert [b["index"] for b in branches] == [0, 1]
    for b in branches:
        assert b["preview_path"] is not None
        assert (tmp_path / "branches" / f"branch_{b['index']}.png").exists()


def test_parallel_node_survives_preview_write_failure(tmp_path):
    """A broken preview must not fail the job — the branch outputs are the payload."""
    from restorax.dag.nodes.parallel import ParallelNode

    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    node = ParallelNode(id="par", name="Parallel", branches=[BranchConfig(name="b0")])
    ctx = _parallel_ctx(tmp_path)
    # A file where the directory should be → mkdir fails inside _write_preview.
    not_a_dir = tmp_path / "occupied"
    not_a_dir.write_text("x")
    ctx.work_dir = not_a_dir

    result = asyncio.run(node.execute(ctx, {"chunks": [[frame]], "meta": None}))
    assert result.outputs["branch_outputs"]
    assert result.metadata["branches"][0]["preview_path"] is None
