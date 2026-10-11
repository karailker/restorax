from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from restorax.core.exceptions import DAGValidationError, PortTypeMismatchError
from restorax.dag.edge import Edge
from restorax.dag.node import Node


@dataclass
class DAG:
    """
    Immutable directed acyclic graph of Nodes connected by Edges.
    Validates structure on construction — raises DAGValidationError on any violation.
    """

    id: str
    name: str
    nodes: dict[str, Node]
    edges: list[Edge]

    def __post_init__(self) -> None:
        self._validate()

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate(self) -> None:
        self._check_node_references()
        self._check_port_names()
        self._check_port_types()
        self._check_no_cycles()

    def _check_node_references(self) -> None:
        for edge in self.edges:
            if edge.source_node_id not in self.nodes:
                raise DAGValidationError(
                    f"Edge references unknown source node '{edge.source_node_id}'"
                )
            if edge.target_node_id not in self.nodes:
                raise DAGValidationError(
                    f"Edge references unknown target node '{edge.target_node_id}'"
                )

    def _check_port_names(self) -> None:
        for edge in self.edges:
            src_node = self.nodes[edge.source_node_id]
            src_ports = {p.name for p in src_node.output_ports}
            if edge.source_port not in src_ports:
                raise DAGValidationError(
                    f"Node '{edge.source_node_id}' has no output port '{edge.source_port}'. "
                    f"Available: {src_ports}"
                )
            tgt_node = self.nodes[edge.target_node_id]
            tgt_ports = {p.name for p in tgt_node.input_ports}
            if edge.target_port not in tgt_ports:
                raise DAGValidationError(
                    f"Node '{edge.target_node_id}' has no input port '{edge.target_port}'. "
                    f"Available: {tgt_ports}"
                )

    def _check_port_types(self) -> None:
        src_port_map: dict[tuple[str, str], type | None] = {}
        for node_id, node in self.nodes.items():
            for port in node.output_ports:
                src_port_map[(node_id, port.name)] = port.type_hint

        for edge in self.edges:
            src_type = src_port_map.get((edge.source_node_id, edge.source_port))
            tgt_node = self.nodes[edge.target_node_id]
            tgt_type = next(
                (p.type_hint for p in tgt_node.input_ports if p.name == edge.target_port), None
            )
            if src_type is not None and tgt_type is not None and src_type != tgt_type:
                raise PortTypeMismatchError(
                    f"Edge {edge.source_node_id}.{edge.source_port} ({src_type.__name__}) "
                    f"→ {edge.target_node_id}.{edge.target_port} ({tgt_type.__name__}): "
                    f"incompatible types"
                )

    def _check_no_cycles(self) -> None:
        """Kahn's algorithm: if topological sort consumes all nodes, graph is acyclic."""
        levels = self._kahn_sort()
        visited = {nid for level in levels for nid in level}
        if len(visited) != len(self.nodes):
            raise DAGValidationError(
                "DAG contains a cycle. Nodes not reachable via topological sort: "
                f"{set(self.nodes) - visited}"
            )

    # ── Topology ──────────────────────────────────────────────────────────────

    def topological_levels(self) -> list[list[str]]:
        """
        Return nodes grouped by level. All nodes in the same level can run
        concurrently (no intra-level dependencies). Level 0 = no upstream deps.
        """
        return self._kahn_sort()

    def _kahn_sort(self) -> list[list[str]]:
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        successors: dict[str, list[str]] = defaultdict(list)

        for edge in self.edges:
            in_degree[edge.target_node_id] += 1
            successors[edge.source_node_id].append(edge.target_node_id)

        queue: deque[str] = deque(nid for nid, deg in in_degree.items() if deg == 0)
        levels: list[list[str]] = []

        while queue:
            level = list(queue)
            queue.clear()
            levels.append(level)
            for nid in level:
                for successor in successors[nid]:
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)

        return levels
