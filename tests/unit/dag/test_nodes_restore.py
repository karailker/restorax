from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from restorax.dag.context import ExecutionContext, ProgressEmitter
from restorax.dag.nodes.restore import RestoreNode


def _make_ctx_for_restore(restorer_mock):
    registry = MagicMock()
    registry.get.return_value = restorer_mock
    emitter = MagicMock(spec=ProgressEmitter)
    return ExecutionContext(
        run_id="r1", job_id="j1", work_dir=Path("/tmp"),
        device=MagicMock(), registry=registry,
        progress_emitter=emitter, logger=MagicMock(),
    )


def test_restore_node_calls_process_frame():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    caps = MagicMock()
    caps.requires_temporal = False
    restorer = MagicMock()
    restorer.capabilities = caps
    restorer.process_frame.return_value = frame

    node = RestoreNode(id="r1", name="R1", restorer_name="real_esrgan")
    ctx = _make_ctx_for_restore(restorer)
    chunks = [[frame, frame]]
    result = asyncio.run(node.execute(ctx, {"chunks": chunks}))

    assert "chunks" in result.outputs
    assert restorer.process_frame.call_count == 2


def test_restore_node_roundtrip():
    node = RestoreNode(id="r1", name="R1", restorer_name="waifu2x", params_dict={"scale": 2})
    data = {"type": "restore", "id": node.id, "name": node.name, **node.to_dict()}
    restored = RestoreNode.from_dict(data)
    assert restored.restorer_name == "waifu2x"
    assert restored.params_dict["scale"] == 2
