"""Vector search node — semantic retrieval from a vector store (F12/F13).

Resolves a query vector (embedding it if given text), runs a top-k cosine
search in the named store/namespace with optional metadata pre-filtering, and
returns ranked hits.

Config::

    {
      "store": "default",
      "namespace": "$.inputs.tenant",
      "query": "$.inputs.question",     # text -> embedded here, OR
      "vector": "$.embed.embedding",    # a precomputed vector
      "top_k": 5,
      "model": "local-hash", "dimension": 256,
      "filter": {"format": "markdown"}  # metadata pre-filter
    }
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.embeddings import DEFAULT_DIMENSION, DEFAULT_MODEL, embed_texts
from app.rag.vectorstore import get_store


@register
class VectorSearchNode(Node):
    type = "vector_search"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        dimension = int(self.config.get("dimension", DEFAULT_DIMENSION))
        model = self.config.get("model", DEFAULT_MODEL)
        vector = self._resolve_vector(ctx, upstream, model, dimension)

        store = get_store(self.config.get("store", "default"), dimension)
        namespace = ctx.resolve(self.config.get("namespace", "default")) or "default"
        hits = store.search(
            vector,
            top_k=int(self.config.get("top_k", 5)),
            namespace=namespace,
            metadata_filter=ctx.resolve(self.config.get("filter")),
        )
        return [h.model_dump() for h in hits]

    def _resolve_vector(self, ctx, upstream, model, dimension) -> list[float]:
        if "vector" in self.config:
            return ctx.resolve(self.config["vector"])
        if "query" in self.config:
            return embed_texts([str(ctx.resolve(self.config["query"]))], model, dimension)[0]
        # Fall back to the upstream output: a precomputed embedding or text.
        source = _single_upstream(upstream) if upstream else ""
        if isinstance(source, dict) and "embedding" in source:
            return source["embedding"]
        return embed_texts([str(source)], model, dimension)[0]
