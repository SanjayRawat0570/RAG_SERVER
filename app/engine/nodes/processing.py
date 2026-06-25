"""Processing node — executes a named operation on its input (F1/F3).

Operations are intentionally generic (text/data transforms) so the
orchestration engine can be exercised end-to-end without external services.
RAG-specific operations (embed, chunk, search) plug in here in later phases.
"""
from __future__ import annotations

from typing import Any, Callable

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.registry import register


def _single_upstream(upstream: dict[str, Any]) -> Any:
    """Return the sole upstream output, or all of them keyed by node id."""
    if len(upstream) == 1:
        return next(iter(upstream.values()))
    return upstream


OPERATIONS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    "identity": lambda v, c: v,
    "uppercase": lambda v, c: str(v).upper(),
    "lowercase": lambda v, c: str(v).lower(),
    "strip": lambda v, c: str(v).strip(),
    "word_count": lambda v, c: len(str(v).split()),
    "char_count": lambda v, c: len(str(v)),
    "prefix": lambda v, c: f"{c.get('value', '')}{v}",
    "suffix": lambda v, c: f"{v}{c.get('value', '')}",
    "multiply": lambda v, c: (v or 0) * c.get("factor", 1),
    "add": lambda v, c: (v or 0) + c.get("amount", 0),
    "set": lambda v, c: c.get("value"),
    "tokenize": lambda v, c: str(v).split(),
    "score": lambda v, c: {"value": v, "score": c.get("score", 0)},
}


@register
class ProcessingNode(Node):
    type = "processing"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        operation = self.config.get("operation", "identity")
        if operation not in OPERATIONS:
            raise ValueError(
                f"Unknown processing operation {operation!r}. "
                f"Available: {sorted(OPERATIONS)}"
            )
        # Allow an explicit input reference; otherwise take the upstream output.
        if "input" in self.config:
            value = ctx.resolve(self.config["input"])
        else:
            value = _single_upstream(upstream)
        return OPERATIONS[operation](value, self.config)
