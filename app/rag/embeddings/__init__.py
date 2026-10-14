"""Multi-model embeddings (F11)."""
from app.rag.embeddings.registry import (
    DEFAULT_DIMENSION,
    DEFAULT_MODEL,
    active_embedding,
    cache_stats,
    clear_cache,
    default_model,
    embed_texts,
    get_embedder,
    register_embedder,
)
from app.rag.embeddings.selector import model_info, select_model

__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_MODEL",
    "active_embedding",
    "cache_stats",
    "clear_cache",
    "default_model",
    "embed_texts",
    "get_embedder",
    "model_info",
    "register_embedder",
    "select_model",
]
