"""Embedder registry + embedding cache (F11).

Models are created by name through factories, so adding a real provider is a
one-liner registration — the rest of the pipeline is model-agnostic.

A small process-wide cache (keyed by model + text hash) avoids re-embedding
identical text, which is the dominant cost at scale.

Built-in models registered at import time
-----------------------------------------
local-hash                            always available (offline, deterministic)
all-MiniLM-L6-v2                      sentence-transformers, 384 dims
all-mpnet-base-v2                     sentence-transformers, 768 dims
paraphrase-multilingual-MiniLM-L12-v2 sentence-transformers, 384 dims (multilingual)
text-embedding-3-small                OpenAI, 512 dims  (API key required)
text-embedding-3-large                OpenAI, 3072 dims (API key required)
embed-english-v3.0                    Cohere, 1024 dims (API key + package)
embed-multilingual-v3.0               Cohere, 1024 dims (API key + package)
"""
from __future__ import annotations

import hashlib
from typing import Callable

from app.rag.embeddings.base import Embedder, HashEmbedder

DEFAULT_MODEL = "local-hash"
DEFAULT_DIMENSION = 256

# name -> factory(dimension) -> Embedder
_FACTORIES: dict[str, Callable[[int], Embedder]] = {
    "local-hash": lambda dim: HashEmbedder(dimension=dim),
}

# Per-model default dimensions (used when caller doesn't specify one)
_MODEL_DIMS: dict[str, int] = {
    "local-hash":                             256,
    "all-MiniLM-L6-v2":                       384,
    "all-mpnet-base-v2":                      768,
    "paraphrase-multilingual-MiniLM-L12-v2":  384,
    "text-embedding-3-small":                 512,
    "text-embedding-3-large":                 3072,
    "embed-english-v3.0":                     1024,
    "embed-multilingual-v3.0":                1024,
    "text-embedding-ada-002":                 1536,
    "embed-english-light-v3.0":               384,
    "embed-multilingual-light-v3.0":          384,
}

_embedders: dict[tuple[str, int], Embedder] = {}
_cache: dict[tuple[str, int, str], list[float]] = {}
_cache_stats: dict[str, int] = {"hits": 0, "misses": 0}


# ── Registration helpers ───────────────────────────────────────────────────────

def register_embedder(name: str, factory: Callable[[int], Embedder]) -> None:
    _FACTORIES[name] = factory


def get_embedder(
    name: str = DEFAULT_MODEL,
    dimension: int | None = None,
) -> Embedder:
    if name not in _FACTORIES:
        raise ValueError(
            f"Unknown embedding model {name!r}. "
            f"Available: {sorted(_FACTORIES)}"
        )
    dim = dimension or _MODEL_DIMS.get(name, DEFAULT_DIMENSION)
    key = (name, dim)
    if key not in _embedders:
        _embedders[key] = _FACTORIES[name](dim)
    return _embedders[key]


# ── Core embedding function with caching ──────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = DEFAULT_MODEL,
    dimension: int | None = None,
) -> list[list[float]]:
    """Embed *texts* with caching for repeated content.

    Cache key: (model, dimension, sha1(text)).  Identical texts are never
    re-sent to the model, regardless of the calling context.
    """
    dim = dimension or _MODEL_DIMS.get(model, DEFAULT_DIMENSION)
    embedder = get_embedder(model, dim)
    results: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    misses: list[int] = []

    for i, text in enumerate(texts):
        ckey = (model, dim, hashlib.sha1(text.encode("utf-8")).hexdigest())
        cached = _cache.get(ckey)
        if cached is not None:
            results[i] = cached
            _cache_stats["hits"] += 1
        else:
            misses.append(i)

    if misses:
        fresh = embedder.embed([texts[i] for i in misses])
        for i, vec in zip(misses, fresh):
            results[i] = vec
            ckey = (model, dim, hashlib.sha1(texts[i].encode("utf-8")).hexdigest())
            _cache[ckey] = vec
            _cache_stats["misses"] += 1

    return results


def clear_cache() -> None:
    """Clear the embedding cache and reset hit/miss counters."""
    _cache.clear()
    _cache_stats["hits"] = 0
    _cache_stats["misses"] = 0


def cache_stats() -> dict:
    """Return cache performance counters."""
    total = _cache_stats["hits"] + _cache_stats["misses"]
    return {
        "entries":  len(_cache),
        "hits":     _cache_stats["hits"],
        "misses":   _cache_stats["misses"],
        "hit_rate": round(_cache_stats["hits"] / total, 4) if total else 0.0,
    }


# ── Auto-register built-in adapters ───────────────────────────────────────────
# Each block is independent: if a library is unavailable, the model simply
# won't appear in _FACTORIES (the registry remains usable for other models).

def _register_sentence_tf() -> None:
    try:
        from app.rag.embeddings.sentence_tf import SentenceTransformerEmbedder
        for model_name in (
            "all-MiniLM-L6-v2",
            "all-mpnet-base-v2",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ):
            _n = model_name
            register_embedder(_n, lambda dim, mn=_n: SentenceTransformerEmbedder(mn))
    except Exception:
        pass


def _register_openai() -> None:
    try:
        import openai as _openai_pkg  # noqa: F401
        from app.rag.embeddings.openai_embed import OpenAIEmbedder
        for model_name, default_dim in (
            ("text-embedding-3-small", 512),
            ("text-embedding-3-large", 3072),
            ("text-embedding-ada-002", 1536),
        ):
            _n, _dd = model_name, default_dim
            register_embedder(_n, lambda d, mn=_n, dd=_dd: OpenAIEmbedder(mn, dimension=d or dd))
    except Exception:
        pass


def _register_cohere() -> None:
    try:
        import cohere as _cohere_pkg  # noqa: F401
        from app.rag.embeddings.cohere_embed import CohereEmbedder
        for model_name, default_dim in (
            ("embed-english-v3.0",            1024),
            ("embed-multilingual-v3.0",       1024),
            ("embed-english-light-v3.0",       384),
            ("embed-multilingual-light-v3.0",  384),
        ):
            _n, _dd = model_name, default_dim
            register_embedder(_n, lambda d, mn=_n, dd=_dd: CohereEmbedder(mn, dimension=d or dd))
    except Exception:
        pass


_register_sentence_tf()
_register_openai()
_register_cohere()
