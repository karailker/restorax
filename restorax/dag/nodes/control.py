from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dataclass
class ChoiceRule:
    field: str                    # key to read from input "meta" dict
    operator: Literal["eq", "gt", "lt", "gte", "lte", "ne"]
    value: Any
    branch_index: int


@dag_node_type("choice")
class ChoiceNode(Node):
    """
    Conditional routing: evaluates rules against input metadata,
    outputs branch_index indicating which downstream path to activate.
    Analogous to AWS Step Functions Choice state.
    """

    def __init__(self, id: str, name: str, rules: list[ChoiceRule] | None = None, default_branch: int = 0, **kwargs: Any) -> None:
        super().__init__(id, name)
        self.rules: list[ChoiceRule] = rules or []
        self.default_branch = default_branch

    @property
    def input_ports(self) -> list[Port]:
        return [Port("meta", object)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("branch_index", int)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        meta = inputs.get("meta") or {}
        meta_dict = meta if isinstance(meta, dict) else vars(meta) if hasattr(meta, "__dict__") else {}

        ops = {
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "lt": lambda a, b: a < b,
            "gte": lambda a, b: a >= b,
            "lte": lambda a, b: a <= b,
        }

        for rule in self.rules:
            field_val = meta_dict.get(rule.field)
            if field_val is not None and ops[rule.operator](field_val, rule.value):
                return NodeResult(outputs={"branch_index": rule.branch_index})

        return NodeResult(outputs={"branch_index": self.default_branch})

    def to_dict(self) -> dict[str, Any]:
        return {
            "rules": [{"field": r.field, "operator": r.operator, "value": r.value, "branch_index": r.branch_index} for r in self.rules],
            "default_branch": self.default_branch,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChoiceNode:
        rules = [ChoiceRule(**r) for r in data.get("rules", [])]
        return cls(id=data["id"], name=data["name"], rules=rules, default_branch=data.get("default_branch", 0))


@dag_node_type("pass")
class PassNode(Node):
    """Identity node. Passes all inputs through as outputs unchanged."""

    @property
    def input_ports(self) -> list[Port]:
        return [Port("data")]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("data")]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        return NodeResult(outputs={"data": inputs.get("data")})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PassNode:
        return cls(id=data["id"], name=data["name"])
