from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from restorax.dag.context import ExecutionContext


@dataclass
class Port:
    name: str
    type_hint: type | None = None


@dataclass
class RetryPolicy:
    max_retries: int = 0
    delay_seconds: float = 1.0
    backoff: Literal["fixed", "exponential"] = "fixed"
    retry_on: tuple[type[Exception], ...] = field(default_factory=lambda: (Exception,))


class NodeState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class NodeResult:
    outputs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0


class Node(ABC):
    """Base class for all DAG nodes. Stateless — all run state lives in DAGRun."""

    def __init__(
        self,
        id: str,
        name: str,
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.id = id
        self.name = name
        self.retry_policy = retry_policy or RetryPolicy()

    @property
    @abstractmethod
    def input_ports(self) -> list[Port]: ...

    @property
    @abstractmethod
    def output_ports(self) -> list[Port]: ...

    @abstractmethod
    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult: ...

    def to_dict(self) -> dict[str, Any]:
        """Serialise node-specific config. Override in subclasses."""
        return {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        """Deserialise from dict. Override in subclasses."""
        return cls(id=data["id"], name=data["name"])
