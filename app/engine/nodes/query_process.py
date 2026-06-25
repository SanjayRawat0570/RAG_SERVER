"""Query process node — query understanding (F13).

Takes a raw query string and returns intent, entities, normalized text, keywords
and an expanded query. Downstream edges typically branch on
``$.<node>.intent`` to pick a search strategy (F2/F5).

Config::

    {"query": "$.inputs.question", "synonyms": {"aes": ["encryption"]}}
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.query import process_query


@register
class QueryProcessNode(Node):
    type = "query_process"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        query = (
            ctx.resolve(self.config["query"])
            if "query" in self.config
            else (_single_upstream(upstream) if upstream else "")
        )
        synonyms = ctx.resolve(self.config.get("synonyms")) if "synonyms" in self.config else None
        return process_query(str(query or ""), synonyms if isinstance(synonyms, dict) else None)
