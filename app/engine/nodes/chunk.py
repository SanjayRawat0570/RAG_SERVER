"""Chunk node — splits a Document into retrieval-ready chunks (F10).

Accepts the upstream :class:`Document` output (or a ``document`` reference, or
plain text) and returns a list of chunk dicts.

Config::

    {
      "strategy": "recursive",   # fixed | recursive | semantic | structure
      "chunk_size": 512,          # in tokens by default
      "overlap": 64,
      "size_unit": "tokens"       # or "chars"
    }
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.chunking import chunk_document
from app.rag.models import Document


@register
class ChunkNode(Node):
    type = "chunk"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        source = (
            ctx.resolve(self.config["document"])
            if "document" in self.config
            else (_single_upstream(upstream) if upstream else None)
        )
        document = self._as_document(source)
        chunks = chunk_document(
            document,
            strategy=self.config.get("strategy", "recursive"),
            config=self.config,
        )
        return [c.model_dump() for c in chunks]

    @staticmethod
    def _as_document(source: Any) -> Document:
        if isinstance(source, Document):
            return source
        if isinstance(source, dict) and "text" in source and "document_id" in source:
            return Document.model_validate(source)
        # Bare text (or anything else) -> wrap in a minimal Document.
        text = source if isinstance(source, str) else str(source or "")
        return Document(document_id="inline", text=text, format="text")
