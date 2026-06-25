"""Context organizer — arrange selected chunks for the prompt (F15).

Strategies
----------
relevance      : highest score first (default)
source         : group chunks from the same source document together
chronological  : sort by date field (newest first)
diversity      : alternate sources to avoid walls of the same document
"""
from __future__ import annotations

from typing import Any


def _date_key(hit: dict[str, Any], date_field: str) -> float:
    val = hit.get("metadata", {}).get(date_field)
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(str(val))
        return dt.timestamp()
    except (ValueError, TypeError):
        return 0.0


def organize_context(
    chunks: list[dict[str, Any]],
    *,
    strategy: str = "relevance",
    source_field: str = "source",
    date_field: str = "date",
) -> list[dict[str, Any]]:
    """Re-order *chunks* for optimal LLM consumption.

    Each returned chunk gets an ``"order"`` field showing its final position.

    Parameters
    ----------
    chunks        : Selected chunks (output of ``select_context``).
    strategy      : "relevance" | "source" | "chronological" | "diversity"
    source_field  : Metadata key for the source/document name.
    date_field    : Metadata key for the document date (chronological only).
    """
    if not chunks:
        return []

    if strategy == "relevance":
        ordered = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

    elif strategy == "source":
        # Group by source, preserve score order within each group.
        groups: dict[str, list] = {}
        for c in chunks:
            src = c.get("metadata", {}).get(source_field, "unknown")
            groups.setdefault(src, []).append(c)
        ordered = []
        for src in sorted(groups, key=lambda s: -max(c.get("score", 0) for c in groups[s])):
            ordered.extend(sorted(groups[src], key=lambda c: c.get("score", 0), reverse=True))

    elif strategy == "chronological":
        ordered = sorted(chunks, key=lambda c: _date_key(c, date_field), reverse=True)

    elif strategy == "diversity":
        # Round-robin across sources.
        groups: dict[str, list] = {}  # type: ignore[assignment]
        for c in chunks:
            src = c.get("metadata", {}).get(source_field, "unknown")
            groups.setdefault(src, []).append(c)
        sources = list(groups.keys())
        ordered = []
        while any(groups[s] for s in sources):
            for src in sources:
                if groups[src]:
                    ordered.append(groups[src].pop(0))

    else:
        ordered = list(chunks)

    for i, c in enumerate(ordered):
        c = dict(c)
        c["order"] = i
        ordered[i] = c

    return ordered


def group_by_source(chunks: list[dict[str, Any]], source_field: str = "source") -> dict[str, list[dict]]:
    """Return {source_name: [chunk, …]} mapping for template rendering."""
    groups: dict[str, list] = {}
    for c in chunks:
        src = c.get("metadata", {}).get(source_field, "Document")
        groups.setdefault(src, []).append(c)
    return groups
