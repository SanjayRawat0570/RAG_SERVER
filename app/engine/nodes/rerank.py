"""Rerank node — refines retrieved candidates (F14).

Takes a list of hits (from search/fusion) and reorders them with a chosen
strategy. The query (for relevance/MMR strategies) comes from ``query`` config
or a ``query_process`` upstream output.

Config::

    {"method": "cross_encoder", "query": "$.qp.normalized", "top_n": 5,
     "semantic_weight": 0.6}
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.rerank import rerank


@register
class RerankNode(Node):
    type = "rerank"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        candidates = (
            ctx.resolve(self.config["candidates"])
            if "candidates" in self.config
            else (_single_upstream(upstream) if upstream else [])
        )
        if not isinstance(candidates, list):
            raise ValueError("rerank node expects a list of candidate hits")

        query = self._resolve_query(ctx)
        method = self.config.get("method", "cross_encoder")
        return rerank(method, query, candidates, self.config)

    def _resolve_query(self, ctx: ExecutionContext) -> str:
        source = ctx.resolve(self.config.get("query", ""))
        if isinstance(source, dict):
            return str(source.get("normalized") or source.get("raw", ""))
        return str(source or "")
