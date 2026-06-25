"""Input node — entry point that exposes the run inputs (F1)."""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.registry import register


@register
class InputNode(Node):
    type = "input"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        # ``key`` selects a single input field; otherwise expose all inputs.
        key = self.config.get("key")
        if key is not None:
            return ctx.inputs.get(key, self.config.get("default"))
        return ctx.inputs
