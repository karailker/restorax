from __future__ import annotations

from typing import Any

from restorax.core.exceptions import DAGValidationError
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node

_NODE_REGISTRY: dict[str, type[Node]] = {}


def dag_node_type(type_id: str):
    """Class decorator that registers a Node subclass under a string type ID."""
    def decorator(cls: type[Node]) -> type[Node]:
        _NODE_REGISTRY[type_id] = cls
        cls._dag_type_id = type_id  # type: ignore[attr-defined]
        return cls
    return decorator


class DAGSerializer:
    """Converts DAG <-> plain dict (JSON-safe). Registered node types only."""

    @staticmethod
    def to_dict(dag: DAG) -> dict[str, Any]:
        return {
            "schema_type": "dag",
            "id": dag.id,
            "name": dag.name,
            "nodes": [
                {
                    "type": getattr(node, "_dag_type_id", type(node).__name__),
                    "id": node.id,
                    "name": node.name,
                    **node.to_dict(),
                }
                for node in dag.nodes.values()
            ],
            "edges": [
                {
                    "source_node_id": e.source_node_id,
                    "source_port": e.source_port,
                    "target_node_id": e.target_node_id,
                    "target_port": e.target_port,
                }
                for e in dag.edges
            ],
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> DAG:
        nodes: dict[str, Node] = {}
        for node_data in data.get("nodes", []):
            type_id = node_data.get("type")
            if type_id not in _NODE_REGISTRY:
                raise DAGValidationError(
                    f"Unknown node type '{type_id}'. "
                    f"Registered types: {list(_NODE_REGISTRY)}"
                )
            cls = _NODE_REGISTRY[type_id]
            nodes[node_data["id"]] = cls.from_dict(node_data)

        edges = [Edge(**e) for e in data.get("edges", [])]
        return DAG(
            id=data["id"],
            name=data["name"],
            nodes=nodes,
            edges=edges,
        )
