"""Entity search node — retrieves documents by named-entity overlap (F4).

Extracts capitalized terms, numbers, and ISO dates from the query, then
scores every record in the namespace by how many of those entities appear
in its stored text. Pairs with ``vector_search`` and ``keyword_search`` for
three-way hybrid retrieval merged via an ``rrf`` node.

Config::

    {
      "store":      "kb",
      "namespace":  "$.inputs.tenant",
      "query":      "$.inputs.question",
      "text_field": "text",
      "top_k":      5,
      "dimension":  256
    }

Output: list of hits — ``{"id", "score", "entity_matches", "matched_entities", "metadata"}``
"""
from __future__ import annotations

import re
from typing import Any

from app.engine.context import ExecutionContext
from app.engine.nodes.base import Node
from app.engine.nodes.processing import _single_upstream
from app.engine.nodes.registry import register
from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.query.processor import extract_entities
from app.rag.vectorstore import get_store

_WS = re.compile(r"\s+")


@register
class EntitySearchNode(Node):
    type = "entity_search"

    async def run(self, ctx: ExecutionContext, upstream: dict[str, Any]) -> Any:
        # ── resolve query ──────────────────────────────────────────────────────
        if "query" in self.config:
            raw = ctx.resolve(self.config["query"])
        else:
            raw = _single_upstream(upstream) if upstream else ""
        query_text = str(raw or "")

        top_k      = int(self.config.get("top_k", 5))
        text_field = self.config.get("text_field", "text")

        # ── extract entities ───────────────────────────────────────────────────
        entities_map = extract_entities(query_text)
        entities: list[str] = (
            entities_map.get("capitalized", [])
            + entities_map.get("numbers", [])
            + entities_map.get("dates", [])
        )
        if not entities:
            return []

        entity_patterns = [re.compile(re.escape(e), re.I) for e in entities]

        # ── score records by entity overlap ───────────────────────────────────
        store     = get_store(
            self.config.get("store", "default"),
            int(self.config.get("dimension", DEFAULT_DIMENSION)),
        )
        namespace = ctx.resolve(self.config.get("namespace", "default")) or "default"
        records   = store.list_records(namespace)
        if not records:
            return []

        scored: list[dict[str, Any]] = []
        for record_id, meta in records:
            text = str(meta.get(text_field, ""))
            matched = [e for e, pat in zip(entities, entity_patterns) if pat.search(text)]
            if not matched:
                continue
            scored.append({
                "id":               record_id,
                "score":            len(matched) / len(entities),
                "entity_matches":   len(matched),
                "matched_entities": matched,
                "metadata":         meta,
            })

        scored.sort(key=lambda h: h["score"], reverse=True)
        return scored[:top_k]
