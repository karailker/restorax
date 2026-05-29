from restorax.core.exceptions import DAGValidationError, NodeExecutionError, PortTypeMismatchError


def test_node_execution_error_carries_cause():
    cause = ValueError("weights missing")
    err = NodeExecutionError("my_node", attempt=2, cause=cause)
    assert "my_node" in str(err)
    assert err.node_id == "my_node"
    assert err.attempt == 2
    assert err.__cause__ is cause
    assert "attempt 2" in str(err)
    assert "weights missing" in str(err)


def test_port_type_mismatch_is_dag_validation_error():
    err = PortTypeMismatchError("port type mismatch")
    assert isinstance(err, DAGValidationError)
