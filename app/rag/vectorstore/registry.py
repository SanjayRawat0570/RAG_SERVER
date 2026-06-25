"""Vector store registry (F12).

Stores are process-wide singletons keyed by name, so data persists across
workflow runs (index in one run, query in the next).

Backend selection
-----------------
``backend="memory"``  — InMemoryVectorStore (FLAT, always available)
``backend="hnsw"``    — HNSWVectorStore (sklearn BallTree, always available)
``backend="supabase"``— SupabaseVectorStore (pgvector, requires credentials)
``backend="auto"``    — "supabase" if configured, else "hnsw"
"""
from __future__ import annotations

from typing import Any

from app.rag.embeddings import DEFAULT_DIMENSION
from app.rag.vectorstore.memory import InMemoryVectorStore

_stores: dict[str, Any] = {}  # name -> store instance


def get_store(
    name: str = "default",
    dimension: int = DEFAULT_DIMENSION,
    backend: str = "memory",
) -> InMemoryVectorStore:
    """Get (or create) a store by *name*.

    If the store already exists under *name*, the existing instance is
    returned regardless of the requested *backend* / *dimension*.  This
    preserves data across requests.
    """
    if name in _stores:
        existing = _stores[name]
        if existing.dimension != dimension:
            raise ValueError(
                f"Store {name!r} exists with dim {existing.dimension}, "
                f"requested {dimension}"
            )
        return existing

    store = _create_store(name, dimension, backend)
    _stores[name] = store
    return store


def _create_store(name: str, dimension: int, backend: str):
    from app.config import settings

    resolved = backend
    if backend == "auto":
        resolved = "supabase" if (settings.supabase_url and settings.supabase_key) else "hnsw"

    if resolved == "supabase":
        try:
            from app.rag.vectorstore.supabase_store import SupabaseVectorStore
            return SupabaseVectorStore(name=name, dimension=dimension)
        except Exception:
            # Credentials missing or client error → fall back to in-memory.
            resolved = "hnsw"

    if resolved == "hnsw":
        from app.rag.vectorstore.indexing import HNSWVectorStore
        return HNSWVectorStore(name=name, dimension=dimension)

    # default: flat memory store
    return InMemoryVectorStore(name=name, dimension=dimension)


def store_stats() -> dict[str, Any]:
    """Return stats for every registered store."""
    return {name: store.stats() for name, store in _stores.items()}


def list_stores() -> list[dict[str, Any]]:
    """Return a list of registered stores with their stats."""
    return [{"name": name, **store.stats()} for name, store in _stores.items()]


def list_namespaces(store_name: str = "default") -> list[dict[str, Any]]:
    """Return all namespaces (partitions) in a store with their vector counts."""
    store = _stores.get(store_name)
    if store is None:
        return []
    # SupabaseVectorStore has its own namespace listing.
    if hasattr(store, "list_namespaces"):
        return store.list_namespaces()
    # InMemoryVectorStore / HNSWVectorStore: iterate _namespaces dict.
    ns_data = getattr(store, "_namespaces", {})
    return [
        {"namespace": ns, "count": len(data.ids)}
        for ns, data in ns_data.items()
    ]


def lookup_store(name: str):
    """Return an existing store by name without any dimension check.

    Returns ``None`` if no store exists with that name.
    Use for read/delete operations where the store was already created.
    """
    return _stores.get(name)


def reset_stores(name: str | None = None) -> None:
    if name is None:
        _stores.clear()
    else:
        _stores.pop(name, None)
