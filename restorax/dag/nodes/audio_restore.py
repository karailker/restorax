"""Audio restoration adapter for the DAG pipeline engine."""

from __future__ import annotations

from typing import Any
import numpy as np

from restorax.audio.restorer import AudioRestorer, AudioRestorerParams
from restorax.core.exceptions import RestorerLoadError
from restorax.dag.node import DAGNode, Port, PortType


class AudioRestoreNode(DAGNode):
    """
    A DAG node that wraps an AudioRestorer (Demucs, VoiceFixer, RNNoise)
    so it can be used inside typed DAG pipelines alongside video restorers.
    """

    def __init__(self, node_id: str, restorer_name: str, params: dict[str, Any] | None = None) -> None:
        super().__init__(node_id)
        self.restorer_name = restorer_name
        self._raw_params = params or {}
        
        # Audio ports: accepts audio, outputs audio
        self.inputs = {"audio": Port("audio", PortType.ANY, description="Audio array (samples, channels)")}
        self.outputs = {"audio": Port("audio", PortType.ANY, description="Processed audio array")}

    async def execute(self, ctx: Any) -> dict[str, Any]:
        audio_data = ctx.get_input(self.id, "audio")
        if audio_data is None:
            raise ValueError(f"AudioRestoreNode '{self.id}' received no audio input")

        registry = ctx.registry
        device = ctx.device
        
        restorer = registry.get(self.restorer_name, device)
        
        if not hasattr(restorer, "process_audio"):
            raise RestorerLoadError(f"Model '{self.restorer_name}' is not a valid AudioRestorer")

        sr = self._raw_params.get("sample_rate", 44100)
        params = AudioRestorerParams(
            sample_rate=sr,
            extra=self._raw_params.get("extra", {}),
        )

        processed = restorer.process_audio(audio_data, params)
        return {"audio": processed}
