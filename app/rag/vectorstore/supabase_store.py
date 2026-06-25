"""Supabase + pgvector store (F12).

Uses the ``vector_chunks`` table with a pgvector ``embedding`` column and an
optional ``match_vectors`` RPC for HNSW-accelerated similarity search.

Required Supabase SQL (run once in the SQL editor)
---------------------------------------------------
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Vector chunks table
CREATE TABLE IF NOT EXISTS vector_chunks (
    id          TEXT PRIMARY KEY,
    store_name  TEXT NOT NULL DEFAULT 'default',
    namespace   TEXT NOT NULL DEFAULT 'default',
    embedding   vector(256),
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- HNSW index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS idx_vector_chunks_hnsw
    ON vector_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Multi-tenant RLS
ALTER TABLE vector_chunks ENABLE ROW LEVEL SECURITY;

-- match_vectors RPC
CREATE OR REPLACE FUNCTION match_vectors(
    query_embedding vector,
    match_namespace TEXT,
    match_store     TEXT,
    match_count     INT
)
RETURNS TABLE (id TEXT, score FLOAT, metadata JSONB)
LANGUAGE sql AS $$
    SELECT id,
           1 - (embedding <=> query_embedding) AS score,
           metadata
    FROM   vector_chunks
    WHERE  namespace = match_namespace
    AND    store_name = match_store
    ORDER  BY embedding <=> query_embedding
    LIMIT  match_count;
$$;

Multi-tenancy
-------------
Pass ``namespace=<tenant_id>`` on every upsert/search/delete call.
Combined with Supabase RLS policies (e.g. ``auth.uid() = tenant_id``),
this gives row-level tenant isolation automatically.
"""
from __future__ import annotations

from typing import Any

from app.rag.vectorstore.base import SearchHit, VectorRecord

_TABLE = "vector_chunks"


class SupabaseVectorStore:
    """Vector store backed by Supabase + pgvector.

    Raises :exc:`RuntimeError` if Supabase credentials are not configured.
    """

    def __init__(
        self,
        name: str = "default",
        dimension: int = 256,
        table: str = _TABLE,
    ) -> None:
        from app.config import settings
        if not settings.supabase_url or not settings.supabase_key:
            raise RuntimeError(
                "Supabase is not configured. "
                "Set SUPABASE_URL and SUPABASE_KEY in your .env file."
            )
        from supabase import create_client  # type: ignore[import]
        self._client = create_client(settings.supabase_url, settings.supabase_key)
        self.name      = name
        self.dimension = dimension
        self._table    = table

    # ── upsert ────────────────────────────────────────────────────────────────

    def upsert(self, records: list[VectorRecord], namespace: str = "default") -> int:
        if not records:
            return 0
        rows = [
            {
                "id":         r.id,
                "store_name": self.name,
                "namespace":  namespace,
                "embedding":  r.vector,
                "metadata":   r.metadata,
            }
            for r in records
        ]
        self._client.table(self._table).upsert(rows).execute()
        return len(rows)

    # ── search ─────────────────────────────────────────────────────────────────

    def search(
        self,
        vector: list[float],
        top_k: int = 5,
        namespace: str = "default",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        # Try RPC (HNSW-accelerated via pgvector index).
        try:
            result = self._client.rpc(
                "match_vectors",
                {
                    "query_embedding": vector,
                    "match_namespace":  namespace,
                    "match_store":      self.name,
                    "match_count":      top_k,
                },
            ).execute()
            hits = [
                SearchHit(
                    id=row["id"],
                    score=float(row["score"]),
                    metadata=row.get("metadata") or {},
                )
                for row in (result.data or [])
            ]
        except Exception:
            # RPC not available — fall back to table scan with client-side scoring.
            hits = self._fallback_search(vector, top_k, namespace)

        # Apply optional metadata filter (post-filter for RPC path).
        if metadata_filter:
            hits = [
                h for h in hits
                if all(h.metadata.get(k) == v for k, v in metadata_filter.items())
            ]

        return hits[:top_k]

    def _fallback_search(
        self,
        vector: list[float],
        top_k: int,
        namespace: str,
    ) -> list[SearchHit]:
        """Fetch all rows and compute cosine similarity in Python (slow but always works)."""
        import numpy as np

        result = (
            self._client.table(self._table)
            .select("id, embedding, metadata")
            .eq("namespace", namespace)
            .eq("store_name", self.name)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return []

        query = np.asarray(vector, dtype=np.float32)
        qn = np.linalg.norm(query)
        if qn > 0:
            query /= qn

        scored: list[tuple[float, str, dict]] = []
        for row in rows:
            emb = row.get("embedding")
            if emb is None:
                continue
            vec = np.asarray(emb, dtype=np.float32)
            vn  = np.linalg.norm(vec)
            if vn > 0:
                vec /= vn
            score = float(np.dot(query, vec))
            scored.append((score, row["id"], row.get("metadata") or {}))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            SearchHit(id=r[1], score=r[0], metadata=r[2])
            for r in scored[:top_k]
        ]

    # ── delete ─────────────────────────────────────────────────────────────────

    def delete(self, ids: list[str], namespace: str = "default") -> int:
        if not ids:
            return 0
        (
            self._client.table(self._table)
            .delete()
            .in_("id", ids)
            .eq("namespace", namespace)
            .eq("store_name", self.name)
            .execute()
        )
        return len(ids)

    # ── count / stats ──────────────────────────────────────────────────────────

    def count(self, namespace: str | None = None) -> int:
        query = (
            self._client.table(self._table)
            .select("id", count="exact")
            .eq("store_name", self.name)
        )
        if namespace is not None:
            query = query.eq("namespace", namespace)
        result = query.execute()
        return getattr(result, "count", 0) or 0

    def stats(self) -> dict[str, Any]:
        total = self.count()
        return {
            "name":      self.name,
            "dimension": self.dimension,
            "index":     "hnsw-pgvector",
            "backend":   "supabase",
            "total":     total,
        }

    def list_namespaces(self) -> list[dict[str, Any]]:
        """Return each namespace with its vector count."""
        result = (
            self._client.table(self._table)
            .select("namespace")
            .eq("store_name", self.name)
            .execute()
        )
        rows = result.data or []
        counts: dict[str, int] = {}
        for row in rows:
            ns = row.get("namespace", "default")
            counts[ns] = counts.get(ns, 0) + 1
        return [{"namespace": ns, "count": c} for ns, c in sorted(counts.items())]
