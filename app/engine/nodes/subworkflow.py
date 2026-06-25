"""Sub-workflow node — runs another workflow as a single node (F1 nested).

The child workflow is embedded in config. ``input_map`` builds the child's
inputs from the parent context (values may be ``$.`` references). The node's
output is the child's ``outputs`` map (or a single output if ``output_key`` is
set).
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.external import _resolve_map
from app.engine.nodes.registry import register
from app.models.workflow import WorkflowDef


@register
class SubWorkflowNode(Node):
    type = "subworkflow"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        # Imported here to avoid a circular import (executor imports nodes).
        from app.engine.executor import WorkflowExecutor

        child = WorkflowDef.model_validate(self.config["workflow"])
        child_inputs = _resolve_map(ctx, self.config.get("input_map", {}))
        result = await WorkflowExecutor(child).run(child_inputs)
        if result.status == "error":
            errors = [r.error for r in result.results if r.status == "error"]
            raise RuntimeError(f"sub-workflow '{child.name}' failed: {errors}")

        output_key = self.config.get("output_key")
        if output_key:
            return result.outputs.get(output_key)
        return result.outputs
