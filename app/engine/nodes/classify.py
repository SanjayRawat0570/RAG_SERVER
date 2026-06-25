"""Classify node — evaluate ordered rules, return best-matching category + confidence (F5).

Unlike switch (which routes downstream edges), classify produces a structured
result that other nodes can inspect. Feed its output into a switch to drive
branching while also recording the confidence score.

Config::

    {
      "input": "$.inputs.filename",        # value to classify (falls back to single upstream)
      "rules": [
        {
          "category":   "pdf",
          "when":       {"left": "$.inputs.filename", "op": "endswith", "right": ".pdf"},
          "confidence": 0.99,              # optional per-rule confidence override
          "label":      "PDF document"     # optional human-readable label for audit
        }
      ],
      "default":            "text",        # category emitted when no rule matches
      "default_confidence": 0.5            # confidence for the default case
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
class ClassifyNode(Node):
    type = "classify"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        if "input" in self.config:
            value = ctx.resolve(self.config["input"])
        else:
            value = _single_upstream(upstream) if upstream else None

        rules: list[dict[str, Any]] = self.config.get("rules", [])
        confidence_map: dict[str, float] = self.config.get("confidence_map", {})
        default_category = self.config.get("default", "unknown")
        default_confidence = float(self.config.get("default_confidence", 0.5))

        for rule in rules:
            when = rule.get("when")
            if when is None or evaluate(when, ctx):
                cat = rule["category"]
                conf = float(
                    rule.get("confidence",
                              confidence_map.get(cat, default_confidence))
                )
                return {
                    "category":     cat,
                    "confidence":   conf,
                    "input":        value,
                    "matched_rule": rule.get("label", cat),
                }

        return {
            "category":     default_category,
            "confidence":   default_confidence,
            "input":        value,
            "matched_rule": "default",
        }
