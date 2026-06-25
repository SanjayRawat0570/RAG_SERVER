"""Output node — produces a final result for the workflow (F1)."""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.external import _resolve_map
from app.engine.nodes.registry import register
from app.engine.nodes.processing import _single_upstream


@register
class OutputNode(Node):
    type = "output"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        # ``value`` may be a single ``$.`` reference, or a dict/list whose nested
        # references are resolved; otherwise pass the upstream output through.
        if "value" in self.config:
            return _resolve_map(ctx, self.config["value"])
        return _single_upstream(upstream) if upstream else None
