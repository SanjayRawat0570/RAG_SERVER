"""Vector store interface and shared types (F12).

A concrete store (in-memory here; Qdrant/Weaviate/Milvus as future adapters)
implements upsert / search / delete over namespaced partitions. Namespaces give
per-tenant isolation (F12 multi-tenancy / F18).
"""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class VectorRecord(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchHit(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorStore(Protocol):
    name: str
    dimension: int

    def upsert(self, records: list[VectorRecord], namespace: str = "default") -> int:
        ...

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        ...

    def delete(self, ids: list[str], namespace: str = "default") -> int:
        ...

    def count(self, namespace: str | None = None) -> int:
        ...

    def stats(self) -> dict[str, Any]:
        ...
