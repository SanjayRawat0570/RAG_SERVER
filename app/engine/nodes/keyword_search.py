"""Keyword search node — BM25 sparse retrieval (F13).

Ranks the records of a vector-store namespace against the query with BM25,
using the namespace's incremental keyword index (kept up to date on every
upsert/delete), so a query does not re-read the corpus. Pairs with
``vector_search`` for hybrid retrieval via an ``rrf`` merge.

Config::

    {"store": "kb", "namespace": "$.inputs.tenant",
     "query": "$.qp.expanded_query", "text_field": "text", "top_k": 5}
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.search import BM25, tokenize
from app.rag.vectorstore import get_store


@register
class KeywordSearchNode(Node):
    type = "keyword_search"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        query_text = self._resolve_query(ctx, upstream)
        text_field = self.config.get("text_field", "text")
        top_k = int(self.config.get("top_k", 5))

        store = get_store(
            self.config.get("store", "default"),
            int(self.config.get("dimension", DEFAULT_DIMENSION)),
        )
        namespace = ctx.resolve(self.config.get("namespace", "default")) or "default"
        if hasattr(store, "keyword_search"):
            return [{"id": h.id, "score": h.score, "metadata": h.metadata}
                    for h in store.keyword_search(query_text, top_k, namespace, text_field)]
        records = store.list_records(namespace)
        if not records:
            return []

        corpus = [tokenize(str(meta.get(text_field, ""))) for _, meta in records]
        bm25 = BM25(corpus)
        scores = bm25.scores(tokenize(query_text))

        ranked = sorted(
            range(len(records)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            {"id": records[i][0], "score": float(scores[i]), "metadata": records[i][1]}
            for i in ranked
            if scores[i] > 0
        ]

    def _resolve_query(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> str:
        source = (
            ctx.resolve(self.config["query"])
            if "query" in self.config
            else (_single_upstream(upstream) if upstream else "")
        )
        if isinstance(source, dict):  # a query_process output
            return str(source.get("expanded_query") or source.get("normalized") or source.get("raw", ""))
        return str(source or "")
