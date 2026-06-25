"""Knowledge graph data models (F21)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


# Entity types recognised by the extractor.
ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "PLACE", "TECHNOLOGY",
    "CONCEPT", "PRODUCT", "DATE", "NUMBER", "OTHER",
}


class Entity(BaseModel):
    id:         str                # "{type}:{name_slug}"
    name:       str
    type:       str = "OTHER"      # one of ENTITY_TYPES
    aliases:    list[str] = Field(default_factory=list)
    doc_ids:    list[str] = Field(default_factory=list)   # source documents
    metadata:   dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class Relation(BaseModel):
    id:         str   # "{subject_id}::{predicate}::{object_id}"
    subject_id: str
    predicate:  str   # e.g. "works_at", "manages", "located_in"
    object_id:  str
    doc_ids:    list[str] = Field(default_factory=list)
    confidence: float = 1.0
    metadata:   dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_now)


class GraphStats(BaseModel):
    entity_count:   int
    relation_count: int
    entity_types:   dict[str, int]   # type → count
    top_entities:   list[dict[str, Any]]  # most-connected entities
