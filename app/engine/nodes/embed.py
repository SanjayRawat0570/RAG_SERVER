"""Embed node — attaches dense vectors to chunks or a query (F11).

Input handling:
* list of chunk dicts  -> each gets an ``embedding`` field (batch embedded)
* a Document/chunk dict -> embeds its ``text`` field
* a plain string        -> returns ``{"text", "embedding"}`` (query embedding)

Config::

    {"model": "local-hash", "dimension": 256, "text_field": "text"}
"""
from __future__ import annotations

from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.embeddings import DEFAULT_DIMENSION, DEFAULT_MODEL, embed_texts


@register
class EmbedNode(Node):
    type = "embed"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        model = self.config.get("model", DEFAULT_MODEL)
        dimension = int(self.config.get("dimension", DEFAULT_DIMENSION))
        field = self.config.get("text_field", "text")

        source = (
            ctx.resolve(self.config["input"])
            if "input" in self.config
            else (_single_upstream(upstream) if upstream else None)
        )

        if isinstance(source, list):
            texts = [str(item.get(field, "")) if isinstance(item, dict) else str(item)
                     for item in source]
            vectors = embed_texts(texts, model, dimension)
            out = []
            for item, vec in zip(source, vectors):
                enriched = dict(item) if isinstance(item, dict) else {field: item}
                enriched["embedding"] = vec
                out.append(enriched)
            return out

        if isinstance(source, dict):
            text = str(source.get(field, ""))
            enriched = dict(source)
            enriched["embedding"] = embed_texts([text], model, dimension)[0]
            return enriched

        text = str(source or "")
        return {"text": text, "embedding": embed_texts([text], model, dimension)[0]}
