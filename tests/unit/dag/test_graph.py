from __future__ import annotations

import pytest

from restorax.core.exceptions import DAGValidationError
from restorax.dag.node import Node, NodeResult, Port, RetryPolicy


class _EchoNode(Node):
    """Test node that echoes its 'data' input to 'data' output."""

    @property
    def input_ports(self):
        return [Port("data")]

    @property
    def output_ports(self):
        return [Port("data")]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"data": inputs.get("data")})


def test_node_has_id_and_name():
    node = _EchoNode(id="n1", name="Echo")
    assert node.id == "n1"
    assert node.name == "Echo"


def test_default_retry_policy():
    node = _EchoNode(id="n1", name="Echo")
    assert node.retry_policy.max_retries == 0


def test_custom_retry_policy():
    policy = RetryPolicy(max_retries=3, delay_seconds=0.5, backoff="exponential")
    node = _EchoNode(id="n1", name="Echo", retry_policy=policy)
    assert node.retry_policy.max_retries == 3
    assert node.retry_policy.backoff == "exponential"


from restorax.core.exceptions import PortTypeMismatchError
from restorax.dag.edge import Edge
from restorax.dag.graph import DAG


class _FrameNode(Node):
    @property
    def input_ports(self):
        return [Port("frames", list)]

    @property
    def output_ports(self):
        return [Port("frames", list)]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"frames": inputs.get("frames", [])})


class _IntNode(Node):
    @property
    def input_ports(self):
        return [Port("value", int)]

    @property
    def output_ports(self):
        return [Port("value", int)]

    async def execute(self, ctx, inputs):
        return NodeResult(outputs={"value": inputs.get("value", 0)})


def _make_linear_dag(n_nodes: int = 2) -> DAG:
    nodes = {f"n{i}": _FrameNode(id=f"n{i}", name=f"Node{i}") for i in range(n_nodes)}
    edges = [
        Edge(
            source_node_id=f"n{i}",
            source_port="frames",
            target_node_id=f"n{i + 1}",
            target_port="frames",
        )
        for i in range(n_nodes - 1)
    ]
    return DAG(id="test", name="Test", nodes=nodes, edges=edges)


def test_valid_dag_builds_without_error():
    dag = _make_linear_dag(3)
    assert len(dag.nodes) == 3


def test_unknown_source_node_raises():
    nodes = {"n0": _EchoNode(id="n0", name="N0")}
    edges = [Edge("ghost", "data", "n0", "data")]
    with pytest.raises(DAGValidationError, match="unknown source node"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_unknown_port_raises():
    nodes = {"n0": _EchoNode(id="n0", name="N0"), "n1": _EchoNode(id="n1", name="N1")}
    edges = [Edge("n0", "nonexistent_port", "n1", "data")]
    with pytest.raises(DAGValidationError, match="no output port"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_cycle_raises():
    nodes = {
        "a": _EchoNode(id="a", name="A"),
        "b": _EchoNode(id="b", name="B"),
    }
    edges = [
        Edge("a", "data", "b", "data"),
        Edge("b", "data", "a", "data"),
    ]
    with pytest.raises(DAGValidationError, match="cycle"):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_port_type_mismatch_raises():
    nodes = {
        "a": _FrameNode(id="a", name="A"),
        "b": _IntNode(id="b", name="B"),
    }
    edges = [Edge("a", "frames", "b", "value")]
    with pytest.raises(PortTypeMismatchError):
        DAG(id="t", name="t", nodes=nodes, edges=edges)


def test_topological_levels_linear():
    dag = _make_linear_dag(3)
    levels = dag.topological_levels()
    assert len(levels) == 3
    assert levels[0] == ["n0"]
    assert levels[1] == ["n1"]
    assert levels[2] == ["n2"]


def test_topological_levels_two_independent_nodes():
    nodes = {
        "a": _EchoNode(id="a", name="A"),
        "b": _EchoNode(id="b", name="B"),
    }
    dag = DAG(id="t", name="t", nodes=nodes, edges=[])
    levels = dag.topological_levels()
    assert len(levels) == 1
    assert set(levels[0]) == {"a", "b"}
