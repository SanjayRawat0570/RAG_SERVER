"""Vector database operations (F12): upsert, indexing, search, namespaces."""
from app.rag.vectorstore.base import SearchHit, VectorRecord, VectorStore
from app.rag.vectorstore.indexing import HNSWVectorStore, index_recommendation, suggest_index_type
from app.rag.vectorstore.memory import InMemoryVectorStore
from app.rag.vectorstore.registry import (
    get_store,
    list_namespaces,
    list_stores,
    lookup_store,
    reset_stores,
    store_stats,
)

__all__ = [
    "SearchHit",
    "VectorRecord",
    "VectorStore",
    "InMemoryVectorStore",
    "HNSWVectorStore",
    "get_store",
    "index_recommendation",
    "list_namespaces",
    "list_stores",
    "lookup_store",
    "reset_stores",
    "store_stats",
    "suggest_index_type",
]
