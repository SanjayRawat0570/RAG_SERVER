"""Switch node — multi-way routing for decision trees (F5).

Where a ``decision`` node forwards a value for edge conditions to inspect, a
``switch`` evaluates an ordered list of cases itself and emits the label of the
first match. Downstream edges then branch on ``$.<switch_id>.case``, which keeps
deep decision trees readable (one switch instead of many parallel conditions).

Config::

    {
      "cases": [
        {"label": "high", "when": {"left": "$.inputs.score", "op": ">",  "right": 0.8}},
        {"label": "mid",  "when": {"left": "$.inputs.score", "op": ">=", "right": 0.5}}
      ],
      "default": "low"
    }
"""
from __future__ import annotations

from typing import Any

from app.engine.conditions import evaluate
from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register


@register
class SwitchNode(Node):
    type = "switch"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        value = ctx.resolve(self.config["input"]) if "input" in self.config else (
            _single_upstream(upstream) if upstream else None
        )
        for case in self.config.get("cases", []):
            if evaluate(case.get("when"), ctx):
                return {"case": case["label"], "value": value}
        return {"case": self.config.get("default", "default"), "value": value}
