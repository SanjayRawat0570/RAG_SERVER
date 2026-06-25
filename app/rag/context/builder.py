"""Assemble retrieved chunks into a context block with citations (F15).

Walks reranked hits in order, formats each with a citation marker and
provenance (heading/source), and includes them until a token budget is hit.
Returns the context text, citation map, and what was included/dropped.
"""
from __future__ import annotations

from typing import Any

from app.rag.models import estimate_tokens


def build_context(hits: list[dict], config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = config or {}
    max_tokens = int(config.get("max_context_tokens", 1024))
    text_field = config.get("text_field", "text")

    parts: list[str] = []
    citations: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    included: list[str] = []
    dropped: list[str] = []
    total = 0

    for hit in hits:
        meta = hit.get("metadata", {})
        text = str(meta.get(text_field, "")).strip()
        if not text:
            continue
        est = estimate_tokens(text)
        # Always include at least one chunk; then respect the budget.
        if included and total + est > max_tokens:
            dropped.append(hit["id"])
            continue
        marker = f"[{len(included) + 1}]"
        heading = meta.get("heading")
        source = meta.get("source")
        label = " ".join(filter(None, [marker, heading, f"({source})" if source else None]))
        parts.append(f"{label}\n{text}")
        citations.append(
            {"marker": marker, "id": hit["id"], "heading": heading, "source": source}
        )
        documents.append(
            {"marker": marker, "id": hit["id"], "heading": heading,
             "source": source, "text": text}
        )
        included.append(hit["id"])
        total += est

    return {
        "context": "\n\n".join(parts),
        "citations": citations,
        "documents": documents,
        "token_estimate": total,
        "included": included,
        "dropped": dropped,
    }
