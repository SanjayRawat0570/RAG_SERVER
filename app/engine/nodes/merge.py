"""Merge node — combines outputs from multiple active branches (F4)."""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.merging import merge
from app.engine.nodes.base import Node
from app.engine.nodes.registry import register


@register
class MergeNode(Node):
    type = "merge"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        strategy = self.config.get("strategy", "concat")
        # Preserve a deterministic order: sort by predecessor node id unless an
        # explicit ``order`` of node ids is provided.
        order = self.config.get("order")
        if order:
            values = [upstream[n] for n in order if n in upstream]
        else:
            values = [upstream[k] for k in sorted(upstream)]
        return merge(strategy, values, self.config)
