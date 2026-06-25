"""Index type selection and HNSW-style vector store (F12).

Three index families
--------------------
FLAT   (< 10 K vectors)   — brute-force cosine; perfect recall, no build cost.
HNSW   (10 K – 1 M)       — sklearn BallTree approximation; sub-linear search.
IVF    (> 1 M vectors)     — IVF concept: cluster centroids + per-cluster search.
                            Not implemented in pure-Python; Supabase/pgvector
                            handles this with CREATE INDEX … USING ivfflat.

In practice, all three stores implement the same VectorStore interface; callers
never need to know which backend is active.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.rag.vectorstore.base import SearchHit, VectorRecord
from app.rag.vectorstore.memory import InMemoryVectorStore, _Namespace, _normalize

# Thresholds for automatic index selection.
FLAT_MAX  =    10_000
HNSW_MAX  = 1_000_000


def suggest_index_type(vector_count: int) -> str:
    """Return the recommended index family for a corpus of *vector_count* vectors."""
    if vector_count < FLAT_MAX:
        return "flat"
    elif vector_count < HNSW_MAX:
        return "hnsw"
    else:
        return "ivf"


def index_recommendation(vector_count: int) -> dict[str, Any]:
    """Return a full recommendation dict with reasons and expected performance."""
    suggested = suggest_index_type(vector_count)
    info = {
        "flat": {
            "reason": "Corpus < 10 K — brute-force is fastest with no build overhead.",
            "recall": "100 %",
            "build_time": "none",
            "search_time": "O(n)",
            "best_for": "development, small datasets, perfect-recall use cases",
        },
        "hnsw": {
            "reason": "Corpus 10 K – 1 M — HNSW gives sub-linear search with high recall.",
            "recall": "~97 %",
            "build_time": "O(n log n)",
            "search_time": "O(log n)",
            "best_for": "production RAG, general-purpose search",
        },
        "ivf": {
            "reason": "Corpus > 1 M — IVF clusters reduce candidates dramatically.",
            "recall": "~90–95 %",
            "build_time": "O(n)",
            "search_time": "O(sqrt(n))",
            "best_for": "large-scale retrieval, Supabase/pgvector with ivfflat index",
        },
    }
    return {
        "vector_count": vector_count,
        "suggested":    suggested,
        "thresholds":   {"flat": FLAT_MAX, "hnsw": HNSW_MAX},
        **info[suggested],
    }


# ── HNSW Namespace (BallTree-backed) ──────────────────────────────────────────

class _HNSWNamespace(_Namespace):
    """Like _Namespace but rebuilds a sklearn BallTree index for fast k-NN search."""

    # Minimum size before we bother building a tree (too small → brute faster).
    _TREE_THRESHOLD = 50

    def __init__(self) -> None:
        super().__init__()
        self._tree = None
        self._dirty = True

    def upsert_one(self, record: VectorRecord) -> None:
        super().upsert_one(record)
        self._dirty = True

    def delete(self, ids: list[str]) -> int:
        removed = super().delete(ids)
        if removed:
            self._dirty = True
        return removed

    def _rebuild(self) -> None:
        if not self._dirty or len(self.ids) < self._TREE_THRESHOLD:
            return
        try:
            from sklearn.neighbors import BallTree  # type: ignore[import]
            matrix = np.vstack(self.vectors)
            self._tree = BallTree(matrix, metric="cosine")
            self._dirty = False
        except Exception:
            self._tree = None

    def search_hnsw(
        self,
        query: np.ndarray,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
    ) -> list[SearchHit]:
        if not self.ids:
            return []

        # With metadata filtering: pre-filter then brute-force (filter reduces candidates).
        if metadata_filter:
            candidates = [
                i for i in range(len(self.ids))
                if all(self.metadata[i].get(k) == v for k, v in metadata_filter.items())
            ]
            if not candidates:
                return []
            sub = np.vstack([self.vectors[i] for i in candidates])
            scores = sub @ query
            order = np.argsort(-scores)[:top_k]
            return [
                SearchHit(
                    id=self.ids[candidates[o]],
                    score=float(scores[o]),
                    metadata=self.metadata[candidates[o]],
                )
                for o in order
            ]

        # Small corpus — brute force beats tree overhead.
        if len(self.ids) < self._TREE_THRESHOLD:
            matrix = np.vstack(self.vectors)
            scores = matrix @ query
            order = np.argsort(-scores)[:top_k]
            return [
                SearchHit(id=self.ids[o], score=float(scores[o]), metadata=self.metadata[o])
                for o in order
            ]

        # Large corpus — use BallTree.
        self._rebuild()
        if self._tree is None:
            # Tree build failed; fall back to brute force.
            matrix = np.vstack(self.vectors)
            scores = matrix @ query
            order = np.argsort(-scores)[:top_k]
            return [
                SearchHit(id=self.ids[o], score=float(scores[o]), metadata=self.metadata[o])
                for o in order
            ]

        k = min(top_k, len(self.ids))
        distances, indices = self._tree.query(query.reshape(1, -1), k=k)
        # BallTree cosine metric returns cosine *distance* (0=identical).
        scores_arr = 1.0 - distances[0]
        return [
            SearchHit(id=self.ids[idx], score=float(sc), metadata=self.metadata[idx])
            for idx, sc in zip(indices[0], scores_arr)
        ]


class HNSWVectorStore(InMemoryVectorStore):
    """In-memory vector store backed by sklearn BallTree for fast approximate k-NN.

    Identical interface to :class:`InMemoryVectorStore`; the search path switches
    to a BallTree index once the namespace reaches ``_HNSWNamespace._TREE_THRESHOLD``
    vectors.  The tree is rebuilt lazily after each upsert/delete batch.

    Index type reported as ``"hnsw-sklearn"`` to distinguish from a native HNSW
    library (hnswlib / faiss) that would be used in production.
    """

    def _ns(self, namespace: str) -> _HNSWNamespace:  # type: ignore[override]
        if namespace not in self._namespaces:
            self._namespaces[namespace] = _HNSWNamespace()
        return self._namespaces[namespace]  # type: ignore[return-value]

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
        query = _normalize(np.asarray(vector, dtype=np.float32))
        return ns.search_hnsw(query, top_k, metadata_filter)  # type: ignore[attr-defined]

    def stats(self) -> dict[str, Any]:
        s = super().stats()
        s["index"] = "hnsw-sklearn"
        return s
