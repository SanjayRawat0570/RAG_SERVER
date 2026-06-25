"""Base class for all node implementations (F1)."""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.models.workflow import NodeDef


class Node:
    """A processing unit in the workflow graph.

    Subclasses implement :meth:`run`, returning this node's output. The output
    is stored in the execution context under the node id and made available to
    downstream nodes.
    """

    type: str = "base"

    def __init__(self, definition: NodeDef) -> None:
        self.definition = definition
        self.id = definition.id
        self.config = definition.config

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        """Execute the node.

        :param ctx: shared run context (inputs + all prior outputs).
        :param upstream: mapping of {predecessor_node_id: output} for the
            predecessors whose edge into this node is active.
        """
        raise NotImplementedError

    def _resolve_config(self, ctx: ExecutionContext, key: str, default: Any = None) -> Any:
        """Resolve a config value that may be a ``$.`` reference."""
        return ctx.resolve(self.config.get(key, default))
