from __future__ import annotations

import pytest
from restorax.dag.node import Node, NodeResult, NodeState, Port, RetryPolicy
from restorax.core.exceptions import DAGValidationError


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
