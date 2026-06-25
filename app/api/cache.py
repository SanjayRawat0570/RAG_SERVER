"""Cache Management API (F17).

Endpoints
---------
GET  /cache/stats                Per-cache hit/miss stats + size
GET  /cache/embedding/stats      Embedding cache stats (from registry)
POST /cache/clear                Clear one or all named caches
POST /cache/invalidate           Invalidate by document ID (clears L1 + semantic stores)
POST /cache/warm                 Pre-populate query cache with a list of questions
GET  /cache/config               Show TTL / threshold settings
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.cache import cache_stats, get_cache, reset_caches
from app.rag.cache.cache import MISS
from app.rag.embeddings.registry import cache_stats as embedding_cache_stats
from app.rag.embeddings.registry import clear_cache as clear_embedding_cache
from app.rag.vectorstore import lookup_store

router = APIRouter(prefix="/cache", tags=["cache"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class ClearRequest(BaseModel):
    name: str | None = None   # None = clear ALL caches


class InvalidateRequest(BaseModel):
    document_id: str = Field(..., min_length=1)
    store:        str = "default"
    namespace:    str = "default"


class WarmRequest(BaseModel):
    queries:   list[str]  = Field(..., min_length=1)
    store:     str        = "default"
    provider:  str        = "stub"
    quality:   str        = "free"
    cache_ttl: float      = 300.0


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/stats")
async def get_stats(_: CurrentUser) -> dict[str, Any]:
    """Return hit/miss stats for every named TTL cache."""
    raw  = cache_stats()
    emb  = embedding_cache_stats()
    total_hits   = sum(v["hits"]   for v in raw.values())
    total_misses = sum(v["misses"] for v in raw.values())
    total        = total_hits + total_misses
    return {
        "caches":          raw,
        "embedding_cache": emb,
        "summary": {
            "total_caches": len(raw),
            "total_hits":   total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": round(total_hits / total, 4) if total else 0.0,
        },
    }


@router.get("/embedding/stats")
async def get_embedding_stats(_: CurrentUser) -> dict[str, Any]:
    """Return stats for the in-process embedding vector cache."""
    return embedding_cache_stats()


@router.post("/clear")
async def clear_cache(req: ClearRequest, _: CurrentUser) -> dict[str, Any]:
    """Clear one named cache or all caches (name=null)."""
    reset_caches(req.name)
    if req.name is None:
        clear_embedding_cache()
    return {
        "cleared": req.name or "all",
        "status":  "ok",
    }


@router.post("/invalidate")
async def invalidate_document(req: InvalidateRequest, _: CurrentUser) -> dict[str, Any]:
    """Invalidate all cache entries related to a document.

    This removes the document's chunks from the vector store AND clears the
    L1 query cache for the affected store, forcing re-retrieval on next query.
    The semantic cache store is also wiped since answers may have changed.
    """
    removed_vectors = 0
    store = lookup_store(req.store)
    if store:
        try:
            count = store.delete(namespace=req.namespace, metadata_filter={"doc_id": req.document_id})
            removed_vectors = count if isinstance(count, int) else 0
        except Exception:
            pass

    # Clear L1 query cache for this store (all keys, since we can't cheaply
    # filter by document).
    cache_name = f"rag-answer:{req.store}"
    reset_caches(cache_name)

    # Clear semantic cache store for this store.
    sem_store_name = f"__semcache__{cache_name}"
    sem_store = lookup_store(sem_store_name)
    if sem_store:
        try:
            sem_store.delete(namespace="default")
        except Exception:
            pass

    return {
        "document_id":    req.document_id,
        "store":          req.store,
        "removed_vectors": removed_vectors,
        "caches_cleared": [cache_name, sem_store_name],
        "status":         "ok",
    }


@router.post("/warm")
async def warm_cache(req: WarmRequest, _: CurrentUser) -> dict[str, Any]:
    """Pre-populate the query cache by running a list of queries now.

    Useful after a deployment or document update so the first real users
    see cached results instead of cold-start latency.
    """
    from app.api.rag import AskRequest, _run_pipeline

    results: list[dict[str, Any]] = []
    for query in req.queries:
        ask_req = AskRequest(
            query=query,
            store=req.store,
            provider=req.provider,
            quality=req.quality,
            cache_ttl=req.cache_ttl,
            use_cache=True,
        )
        try:
            result = await _run_pipeline(ask_req)
            results.append({
                "query":      query,
                "cached":     result.get("cache_hit", False),
                "provider":   result.get("provider"),
                "answer_len": len(result.get("answer", "")),
                "status":     "ok",
            })
        except Exception as exc:
            results.append({"query": query, "status": "error", "detail": str(exc)})

    warmed = sum(1 for r in results if r.get("status") == "ok")
    return {
        "queries_requested": len(req.queries),
        "queries_warmed":    warmed,
        "results":           results,
    }


@router.get("/config")
async def get_cache_config(_: CurrentUser) -> dict[str, Any]:
    """Return default cache configuration values."""
    return {
        "l1_exact": {
            "type":       "TTLCache (LRU + TTL)",
            "default_ttl": 300,
            "default_maxsize": 1024,
            "key":        "SHA1(query + store + provider + quality)",
            "scope":      "per-store",
        },
        "l2_semantic": {
            "type":             "SemanticCache (cosine similarity)",
            "default_threshold": 0.95,
            "description":      "Near-duplicate queries reuse cached answers",
            "backend":          "InMemoryVectorStore (flat cosine)",
        },
        "l3_embedding": {
            "type":        "In-process dict (SHA1 keyed)",
            "description": "Avoids re-embedding identical text strings",
            "scope":       "process-wide, all models",
        },
    }
