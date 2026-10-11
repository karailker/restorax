from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Edge:
    source_node_id: str
    source_port: str
    target_node_id: str
    target_port: str
