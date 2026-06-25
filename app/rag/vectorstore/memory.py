"""In-process vector store with brute-force (FLAT) cosine search (F12).

Backed by NumPy: each namespace keeps a matrix of L2-normalized vectors plus
aligned id/metadata lists, so search is a single matrix-vector product. This is
the FLAT index; an HNSW/IVF adapter would implement the same interface for
larger corpora. Upsert is update-or-insert by id; metadata filtering pre-filters
candidates before scoring.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.rag.vectorstore.base import SearchHit, VectorRecord


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


class _Namespace:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.metadata: list[dict[str, Any]] = []
        self._pos: dict[str, int] = {}

    def upsert_one(self, record: VectorRecord) -> None:
        vec = _normalize(np.asarray(record.vector, dtype=np.float32))
        if record.id in self._pos:  # update in place
            i = self._pos[record.id]
            self.vectors[i] = vec
            self.metadata[i] = record.metadata
        else:  # insert
            self._pos[record.id] = len(self.ids)
            self.ids.append(record.id)
            self.vectors.append(vec)
            self.metadata.append(record.metadata)

    def delete(self, ids: list[str]) -> int:
        removed = 0
        for rid in ids:
            if rid in self._pos:
                i = self._pos.pop(rid)
                self.ids.pop(i)
                self.vectors.pop(i)
                self.metadata.pop(i)
                removed += 1
                # Reindex positions after the removed slot.
                for j in range(i, len(self.ids)):
                    self._pos[self.ids[j]] = j
        return removed


class InMemoryVectorStore:
    def __init__(self, name: str, dimension: int) -> None:
        self.name = name
        self.dimension = dimension
        self._namespaces: dict[str, _Namespace] = {}

    def _ns(self, namespace: str) -> _Namespace:
        return self._namespaces.setdefault(namespace, _Namespace())

    def upsert(self, records: list[VectorRecord], namespace: str = "default") -> int:
        ns = self._ns(namespace)
        for record in records:
            if len(record.vector) != self.dimension:
                raise ValueError(
                    f"Vector dim {len(record.vector)} != store dim {self.dimension}"
                )
            ns.upsert_one(record)
        return len(records)

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        ns = self._namespaces.get(namespace)
        if not ns or not ns.ids:
            return []

        candidates = range(len(ns.ids))
        if metadata_filter:
            candidates = [
                i for i in candidates
                if all(ns.metadata[i].get(k) == v for k, v in metadata_filter.items())
            ]
            if not candidates:
                return []

        query = _normalize(np.asarray(vector, dtype=np.float32))
        idx = list(candidates)
        matrix = np.vstack([ns.vectors[i] for i in idx])
        scores = matrix @ query  # cosine (all rows + query are normalized)

        order = np.argsort(-scores)[:top_k]
        return [
            SearchHit(id=ns.ids[idx[o]], score=float(scores[o]), metadata=ns.metadata[idx[o]])
            for o in order
        ]

    def delete(self, ids: list[str], namespace: str = "default") -> int:
        ns = self._namespaces.get(namespace)
        return ns.delete(ids) if ns else 0

    def list_records(self, namespace: str = "default") -> list[tuple[str, dict[str, Any]]]:
        """All (id, metadata) in a namespace — the corpus for keyword search (F13)."""
        ns = self._namespaces.get(namespace)
        if not ns:
            return []
        return [(ns.ids[i], ns.metadata[i]) for i in range(len(ns.ids))]

    def count(self, namespace: str | None = None) -> int:
        if namespace is not None:
            ns = self._namespaces.get(namespace)
            return len(ns.ids) if ns else 0
        return sum(len(ns.ids) for ns in self._namespaces.values())

    def stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "index": "flat-cosine",
            "namespaces": {ns: len(data.ids) for ns, data in self._namespaces.items()},
            "total": self.count(),
        }
