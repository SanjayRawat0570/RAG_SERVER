"""Semantic response cache (F17).

Caches answers keyed by the *meaning* of a query: a near-duplicate question
(cosine similarity ≥ threshold) reuses a previous answer, cutting repeat LLM
calls. Implemented on top of the existing vector store — the cached payload is
stored as the record metadata, so it inherits namespacing and FLAT search.

Entries expire after ``ttl`` seconds, a cache keeps at most ``max_entries``
(oldest evicted), and an entry written before the tenant's knowledge base last
changed is never served: every write to a tenant's documents bumps
:func:`kb_version` (called by the vector store), so answers cannot outlive the
text they were built from.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.vectorstore import VectorRecord, get_store

_kb_versions: dict[str, int] = {}


def bump_kb_version(namespace: str) -> None:
    """A tenant's documents changed (called on every upsert/delete)."""
    tenant = (namespace or "default").split("/", 1)[0]
    _kb_versions[tenant] = _kb_versions.get(tenant, 0) + 1


def kb_version(tenant: str | None) -> int:
    return _kb_versions.get(tenant or "default", 0)


class SemanticCache:
    def __init__(self, name: str, dimension: int = DEFAULT_DIMENSION, threshold: float = 0.95,
                 *, ttl: float | None = None, max_entries: int | None = None,
                 tenant: str | None = None) -> None:
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.tenant = tenant
        # Prefix keeps cache vectors isolated from real document stores.
        self.store = get_store(f"__semcache__{name}", dimension)

    def lookup_scored(self, vector: list[float]) -> tuple[Any | None, float | None]:
        """(payload, similarity) of the closest live entry, payload None below threshold."""
        hits = self.store.search(vector, top_k=1)
        if not hits:
            return None, None
        hit = hits[0]
        md = hit.metadata
        expired = self.ttl is not None and time.time() - md.get("ts", 0) > self.ttl
        stale = self.tenant is not None and md.get("kb_version") != kb_version(self.tenant)
        if hit.score >= self.threshold and not expired and not stale:
            return md.get("payload"), hit.score
        return None, hit.score

    def lookup(self, vector: list[float]) -> Any | None:
        return self.lookup_scored(vector)[0]

    def put(self, key_text: str, vector: list[float], payload: Any) -> None:
        rid = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:16]
        self.store.upsert([VectorRecord(id=rid, vector=vector, metadata={
            "payload": payload, "ts": time.time(), "kb_version": kb_version(self.tenant)})])
        if self.max_entries and self.store.count("default") > self.max_entries:
            records = sorted(self.store.list_records("default"), key=lambda r: r[1].get("ts", 0))
            self.store.delete([rid for rid, _ in records[: len(records) - self.max_entries]])
