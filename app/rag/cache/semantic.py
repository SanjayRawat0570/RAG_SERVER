"""Semantic response cache (F17).

Caches answers keyed by the *meaning* of a query: a near-duplicate question
(cosine similarity ≥ threshold) reuses a previous answer, cutting repeat LLM
calls. Implemented on top of the existing vector store — the cached payload is
stored as the record metadata, so it inherits namespacing and FLAT search.
"""
from __future__ import annotations

import hashlib
from typing import Any

from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.vectorstore import VectorRecord, get_store


class SemanticCache:
    def __init__(self, name: str, dimension: int = DEFAULT_DIMENSION, threshold: float = 0.95) -> None:
        self.threshold = threshold
        # Prefix keeps cache vectors isolated from real document stores.
        self.store = get_store(f"__semcache__{name}", dimension)

    def lookup(self, vector: list[float]) -> Any | None:
        hits = self.store.search(vector, top_k=1)
        if hits and hits[0].score >= self.threshold:
            return hits[0].metadata.get("payload")
        return None

    def put(self, key_text: str, vector: list[float], payload: Any) -> None:
        rid = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:16]
        self.store.upsert([VectorRecord(id=rid, vector=vector, metadata={"payload": payload})])
