"""Decision node — passes its input through (F2/F5).

Branching itself is driven by *edge conditions* evaluated by the executor, so a
decision node mostly forwards a value that downstream edge conditions inspect.
For convenience it can also pre-compute and expose named condition results.
"""
from __future__ import annotations

from typing import Any

from app.engine.conditions import evaluate
from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.registry import register
from app.engine.nodes.processing import _single_upstream


@register
class DecisionNode(Node):
    type = "decision"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        if "input" in self.config:
            value = ctx.resolve(self.config["input"])
        else:
            value = _single_upstream(upstream)

        # Optional: evaluate named conditions and attach the results so that
        # downstream edges can branch on ``$.<decision_id>.<name>``.
        checks = self.config.get("checks")
        if checks:
            result = {"value": value}
            for name, cond in checks.items():
                result[name] = evaluate(cond, ctx)
            return result
        return value
