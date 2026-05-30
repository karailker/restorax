from __future__ import annotations

from typing import Any

from restorax.dag.context import ExecutionContext
from restorax.dag.node import Node, NodeResult, Port
from restorax.dag.serializer import dag_node_type


@dag_node_type("map")
class MapNode(Node):
    """
    Apply a sub-DAG to each item in a list sequentially.
    Analogous to AWS Step Functions Map state.
    Useful for batch-processing multiple video clips with the same pipeline.
    """

    def __init__(self, id: str, name: str, sub_dag_dict: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(id, name)
        self.sub_dag_dict: dict[str, Any] = sub_dag_dict or {}

    @property
    def input_ports(self) -> list[Port]:
        return [Port("items", list)]

    @property
    def output_ports(self) -> list[Port]:
        return [Port("results", list)]

    async def execute(self, ctx: ExecutionContext, inputs: dict[str, Any]) -> NodeResult:
        from restorax.dag.executor import DAGExecutor
        from restorax.dag.serializer import DAGSerializer

        items: list[Any] = inputs["items"]
        results: list[Any] = []

        if not self.sub_dag_dict:
            return NodeResult(outputs={"results": items})

        sub_dag = DAGSerializer.from_dict(self.sub_dag_dict)

        for i, item in enumerate(items):
            item_ctx = ExecutionContext(
                run_id=f"{ctx.run_id}-map-{i}",
                job_id=ctx.job_id,
                work_dir=ctx.work_dir / f"map_{i}",
                device=ctx.device,
                registry=ctx.registry,
                progress_emitter=ctx.progress_emitter,
                logger=ctx.logger,
                config=ctx.config,
            )
            item_ctx.work_dir.mkdir(parents=True, exist_ok=True)
            root_node_id = sub_dag.topological_levels()[0][0]
            run = await DAGExecutor().execute(
                sub_dag, item_ctx, initial_inputs={root_node_id: {"data": item}}
            )
            last_node_id = sub_dag.topological_levels()[-1][0]
            last_result = run.node_results.get(last_node_id)
            results.append(last_result.outputs.get("data") if last_result else None)
            ctx.progress_emitter.emit(self.id, (i + 1) / max(len(items), 1))

        return NodeResult(outputs={"results": results})

    def to_dict(self) -> dict[str, Any]:
        return {"sub_dag_dict": self.sub_dag_dict}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MapNode:
        return cls(id=data["id"], name=data["name"], sub_dag_dict=data.get("sub_dag_dict"))
