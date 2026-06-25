"""Upsert node — writes embedded records into a vector store (F12).

Takes a list of embedded items (chunks with an ``embedding`` field) and upserts
them into a named store + namespace. The chunk text and metadata are stored
alongside the vector so retrieval can return content without a second lookup.

Config::

    {
      "store": "default",
      "namespace": "$.inputs.tenant",   # per-tenant isolation
      "dimension": 256,
      "id_field": "chunk_id"
    }
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.vectorstore import VectorRecord, get_store


@register
class UpsertNode(Node):
    type = "upsert"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        items = (
            ctx.resolve(self.config["input"])
            if "input" in self.config
            else _single_upstream(upstream)
        )
        if isinstance(items, dict):
            items = [items]
        if not isinstance(items, list):
            raise ValueError("upsert node expects a list of embedded records")

        dimension = int(self.config.get("dimension", DEFAULT_DIMENSION))
        store_name = self.config.get("store", "default")
        namespace = ctx.resolve(self.config.get("namespace", "default")) or "default"
        id_field = self.config.get("id_field", "chunk_id")
        store = get_store(store_name, dimension)

        records: list[VectorRecord] = []
        for i, item in enumerate(items):
            if "embedding" not in item:
                raise ValueError("each record must have an 'embedding' (run embed first)")
            # Flatten a chunk's nested ``metadata`` (heading, title, …) up to the
            # top level so retrieval can read fields directly (e.g. hit.metadata.heading).
            flat = {k: v for k, v in item.items() if k not in ("embedding", "metadata")}
            metadata = {**item.get("metadata", {}), **flat}
            records.append(
                VectorRecord(
                    id=str(item.get(id_field, f"{namespace}:{i}")),
                    vector=item["embedding"],
                    metadata=metadata,
                )
            )

        upserted = store.upsert(records, namespace=namespace)
        return {
            "store": store_name,
            "namespace": namespace,
            "upserted": upserted,
            "namespace_count": store.count(namespace),
        }
