"""Augment node — build context + final prompt from retrieved hits (F15).

Combines context assembly (budgeting + citations) and prompt engineering into a
single output ready for the F16 LLM node: ``messages``, ``prompt``, ``context``,
and ``citations``.

Config::

    {
      "query": "$.qp.normalized",
      "max_context_tokens": 1024,
      "chain_of_thought": false,
      "answer_format": "A short paragraph with citations.",
      "system": "..."          # optional system prompt override
      "template": "..."        # optional custom user template ({context}/{question})
    }
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.context import build_context, build_prompt


@register
class AugmentNode(Node):
    type = "augment"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        hits = (
            ctx.resolve(self.config["hits"])
            if "hits" in self.config
            else (_single_upstream(upstream) if upstream else [])
        )
        if not isinstance(hits, list):
            raise ValueError("augment node expects a list of retrieved hits")

        query = self._resolve_query(ctx)
        ctx_block = build_context(hits, self.config)
        prompt = build_prompt(query, ctx_block["context"], self.config)

        return {
            "query": query,
            "system": prompt["system"],
            "prompt": prompt["prompt"],
            "messages": prompt["messages"],
            "context": ctx_block["context"],
            "citations": ctx_block["citations"],
            "documents": ctx_block["documents"],
            "token_estimate": ctx_block["token_estimate"],
            "included": ctx_block["included"],
            "dropped": ctx_block["dropped"],
        }

    def _resolve_query(self, ctx: ExecutionContext) -> str:
        source = ctx.resolve(self.config.get("query", ""))
        if isinstance(source, dict):
            return str(source.get("normalized") or source.get("raw", ""))
        return str(source or "")
