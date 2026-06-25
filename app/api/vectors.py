"""F12: Vector Database Operations — HTTP API.

Endpoints
---------
POST   /vectors/upsert               Store embeddings (update-or-insert)
POST   /vectors/search               Similarity search with optional metadata filter
DELETE /vectors/records              Remove vectors by ID
GET    /vectors/stores               List all active stores with stats
GET    /vectors/stores/{name}/ns     List namespaces / partitions for a store
POST   /vectors/index/suggest        Recommend index type for a corpus size
POST   /vectors/maintenance/{name}   Maintenance plan for a store
POST   /vectors/upsert-chunks        Embed chunks and upsert in one call
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.rag.embeddings import embed_texts
from app.rag.embeddings.registry import _MODEL_DIMS
from app.rag.vectorstore import (
    SearchHit,
    VectorRecord,
    get_store,
    index_recommendation,
    list_namespaces,
    list_stores,
    lookup_store,
    reset_stores,
    store_stats,
    suggest_index_type,
)

router = APIRouter(prefix="/vectors", tags=["vectors"])


# ── Request models ─────────────────────────────────────────────────────────────

class RecordIn(BaseModel):
    id: str
    vector: list[float]
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertRequest(BaseModel):
    records:   list[RecordIn]
    store:     str = "default"
    namespace: str = "default"
    backend:   str = "memory"   # "memory" | "hnsw" | "supabase" | "auto"


class SearchRequest(BaseModel):
    vector:    list[float]
    top_k:     int = Field(default=5, ge=1, le=100)
    store:     str = "default"
    namespace: str = "default"
    filters:   dict[str, Any] | None = None


class DeleteRequest(BaseModel):
    ids:       list[str]
    store:     str = "default"
    namespace: str = "default"


class IndexSuggestRequest(BaseModel):
    vector_count: int = Field(..., ge=0)
    preference:   str = "balanced"   # "recall" | "speed" | "balanced"


class UpsertChunksRequest(BaseModel):
    chunks:    list[dict[str, Any]]  # [{text: ..., metadata: ...}]
    model:     str = "local-hash"
    store:     str = "default"
    namespace: str = "default"
    backend:   str = "memory"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_or_create(store_name: str, dimension: int, backend: str):
    try:
        return get_store(store_name, dimension, backend)
    except RuntimeError as exc:
        raise HTTPException(424, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upsert")
async def upsert_vectors(
    request: UpsertRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Store embeddings in the vector database.

    Upsert semantics: if a record with the same ``id`` already exists in the
    namespace, its vector and metadata are **replaced**.  If not, a new record
    is inserted.  This prevents duplicates when re-indexing a document.

    Use ``namespace`` to partition by tenant, document type, or date range.
    Use ``backend`` to choose ``"memory"`` (dev), ``"hnsw"`` (medium scale),
    or ``"supabase"`` (production pgvector).
    """
    if not request.records:
        raise HTTPException(422, "No records to upsert.")

    dim = len(request.records[0].vector)
    store = _get_or_create(request.store, dim, request.backend)

    records = [VectorRecord(id=r.id, vector=r.vector, metadata=r.metadata)
               for r in request.records]
    n = store.upsert(records, namespace=request.namespace)

    return {
        "upserted":  n,
        "store":     request.store,
        "namespace": request.namespace,
        "backend":   store.stats().get("index", "flat-cosine"),
        "total":     store.count(),
    }


@router.post("/search")
async def search_vectors(
    request: SearchRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Run a similarity search against the vector store.

    Returns the ``top_k`` closest vectors ranked by cosine similarity score
    (1 = identical, 0 = orthogonal, -1 = opposite).

    Use ``filters`` to restrict results to records whose metadata matches
    all key-value pairs in the filter dict (exact-match AND logic).
    """
    store = lookup_store(request.store)
    if store is None:
        raise HTTPException(404, f"Store '{request.store}' not found.")

    hits = store.search(
        request.vector,
        top_k=request.top_k,
        namespace=request.namespace,
        metadata_filter=request.filters,
    )
    return {
        "hits":      [h.model_dump() for h in hits],
        "count":     len(hits),
        "store":     request.store,
        "namespace": request.namespace,
    }


@router.delete("/records")
async def delete_vectors(
    request: DeleteRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Delete vectors by their IDs.

    Deletions are scoped to the ``namespace`` — the same ID in a different
    namespace is not affected.
    """
    store = lookup_store(request.store)
    if store is None:
        raise HTTPException(404, f"Store '{request.store}' not found.")

    removed = store.delete(request.ids, namespace=request.namespace)
    return {
        "deleted":   removed,
        "requested": len(request.ids),
        "store":     request.store,
        "namespace": request.namespace,
    }


@router.get("/stores")
async def list_vector_stores(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    List all active vector stores with their statistics.

    Shows store name, dimension, index type (flat/hnsw/pgvector), total vector
    count, and a per-namespace breakdown.
    """
    stores = list_stores()
    return {
        "stores":        stores,
        "count":         len(stores),
        "total_vectors": sum(s.get("total", 0) for s in stores),
    }


@router.get("/stores/{store_name}/ns")
async def list_store_namespaces(
    store_name: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    List all namespaces (partitions) inside a store with their vector counts.

    Namespaces are used for:
    - **Multi-tenancy**: one namespace per user/organisation
    - **Document types**: separate PDFs from spreadsheets
    - **Date ranges**: partition by ingestion month for time-scoped searches
    """
    namespaces = list_namespaces(store_name)
    total = sum(ns["count"] for ns in namespaces)
    return {
        "store":      store_name,
        "namespaces": namespaces,
        "count":      len(namespaces),
        "total":      total,
    }


@router.post("/index/suggest")
async def suggest_index(
    request: IndexSuggestRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Recommend the best vector index type for a given corpus size.

    Index families
    --------------
    **FLAT**   (< 10 K)   — brute-force, 100 % recall, no build time
    **HNSW**   (10 K–1 M) — approximate, ~97 % recall, sub-linear search
    **IVF**    (> 1 M)    — cluster-based, ~92 % recall, fastest at scale

    The recommendation adjusts for the ``preference`` parameter:
    - ``"recall"`` — bias toward exact or near-exact results
    - ``"speed"``  — bias toward lowest latency
    - ``"balanced"``— default trade-off
    """
    rec = index_recommendation(request.vector_count)

    # Adjust for preference
    if request.preference == "recall" and rec["suggested"] == "ivf":
        rec["suggested"] = "hnsw"
        rec["note"] = "Switched from IVF to HNSW to prioritise recall over speed."
    elif request.preference == "speed" and rec["suggested"] == "flat":
        rec["note"] = "FLAT is already optimal at this scale — no change needed."

    return rec


@router.post("/maintenance/{store_name}")
async def maintenance_plan(
    store_name: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Return a maintenance plan for the specified store.

    Maintenance tasks:
    - **Reindex**: rebuild index after many deletes or dimension changes
    - **Compact**: remove tombstoned entries and defragment memory
    - **Analyse**: update statistics for query planning
    - **Vacuum**: reclaim storage (Supabase/pgvector only)

    This endpoint returns the plan without executing it — useful for
    understanding what maintenance is needed before scheduling it.
    """
    store = lookup_store(store_name)
    if store is None:
        raise HTTPException(404, f"Store '{store_name}' not found.")

    stats = store.stats()
    total = stats.get("total", 0)
    index = stats.get("index", "flat-cosine")
    ns_count = len(getattr(store, "_namespaces", {}))

    actions: list[dict[str, Any]] = []

    # Suggest reindex if large store on flat index
    if total >= 10_000 and "flat" in index:
        actions.append({
            "action": "reindex",
            "reason": f"Store has {total:,} vectors but uses FLAT index — upgrade to HNSW.",
            "command": f"get_store('{store_name}', backend='hnsw')",
            "priority": "high",
        })

    # Suggest compact if many namespaces
    if ns_count > 20:
        actions.append({
            "action": "compact",
            "reason": f"{ns_count} namespaces detected — merge or archive stale ones.",
            "priority": "medium",
        })

    # Always suggest analyse for stats freshness
    actions.append({
        "action": "analyse",
        "reason": "Refresh query statistics for optimal search performance.",
        "priority": "low",
        "schedule": "weekly",
    })

    if "pgvector" in index or "supabase" in stats.get("backend", ""):
        actions.append({
            "action": "vacuum",
            "reason": "Reclaim storage from deleted rows in pgvector table.",
            "sql": f"VACUUM ANALYZE {store_name};",
            "priority": "low",
            "schedule": "monthly",
        })

    suggested_index = suggest_index_type(total)
    return {
        "store":           store_name,
        "total_vectors":   total,
        "current_index":   index,
        "suggested_index": suggested_index,
        "needs_reindex":   suggested_index not in index,
        "actions":         actions,
        "namespaces":      ns_count,
        "generated_at":    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


@router.post("/upsert-chunks")
async def upsert_chunks(
    request: UpsertChunksRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Embed chunks and store them in the vector database in one request.

    Each chunk must have a ``text`` field; any other fields are stored as
    metadata.  The embedding model is called in batch (one API call per request)
    so this is significantly cheaper than embedding + upsert separately.
    """
    if not request.chunks:
        raise HTTPException(422, "No chunks provided.")

    texts = [c.get("text", "") for c in request.chunks]
    dim   = _MODEL_DIMS.get(request.model, 256)
    try:
        vectors = embed_texts(texts, request.model, dim)
    except RuntimeError as exc:
        raise HTTPException(424, str(exc)) from exc

    records = [
        VectorRecord(
            id=c.get("chunk_id", f"chunk-{i}"),
            vector=vectors[i],
            metadata={k: v for k, v in c.items() if k not in ("text",)},
        )
        for i, c in enumerate(request.chunks)
    ]

    store = _get_or_create(request.store, dim, request.backend)
    n = store.upsert(records, namespace=request.namespace)

    return {
        "upserted":  n,
        "model":     request.model,
        "dimension": dim,
        "store":     request.store,
        "namespace": request.namespace,
        "total":     store.count(),
    }
