from __future__ import annotations

import pytest

from restorax.core.exceptions import DAGValidationError
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import DAGSerializer, dag_node_type


@dag_node_type("_test_echo")
class _SerEchoNode(Node):
    def __init__(self, id: str, name: str, label: str = "", **kwargs):
        super().__init__(id, name)
        self.label = label

    @property
    def input_ports(self):
        return [Port("data")]

    @property
    def output_ports(self):
        return [Port("data")]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"data": inputs.get("data")})

    def to_dict(self):
        return {"label": self.label}

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], name=data["name"], label=data.get("label", ""))


def _make_two_node_dag() -> DAG:
    nodes = {
        "src": _SerEchoNode(id="src", name="Source", label="hello"),
        "dst": _SerEchoNode(id="dst", name="Dest", label="world"),
    }
    edges = [Edge("src", "data", "dst", "data")]
    return DAG(id="ser-test", name="Serialization Test", nodes=nodes, edges=edges)


def test_roundtrip_preserves_structure():
    dag = _make_two_node_dag()
    data = DAGSerializer.to_dict(dag)
    restored = DAGSerializer.from_dict(data)

    assert restored.id == dag.id
    assert restored.name == dag.name
    assert set(restored.nodes) == {"src", "dst"}
    assert restored.nodes["src"].label == "hello"  # type: ignore[attr-defined]
    assert len(restored.edges) == 1
    assert restored.edges[0].source_node_id == "src"


def test_schema_type_is_dag():
    data = DAGSerializer.to_dict(_make_two_node_dag())
    assert data["schema_type"] == "dag"


def test_unknown_node_type_raises():
    data = {
        "id": "t", "name": "t",
        "nodes": [{"type": "totally_unknown", "id": "x", "name": "X"}],
        "edges": [],
    }
    with pytest.raises(DAGValidationError, match="Unknown node type"):
        DAGSerializer.from_dict(data)
