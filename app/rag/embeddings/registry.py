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
bge-small-en-v1.5                     fastembed/ONNX, 384 dims (local, offline after download)
text-embedding-3-small                OpenAI, 512 dims  (API key required)
text-embedding-3-large                OpenAI, 3072 dims (API key required)
embed-english-v3.0                    Cohere, 1024 dims (API key + package)
embed-multilingual-v3.0               Cohere, 1024 dims (API key + package)
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache
from typing import Callable

from app.rag.embeddings.base import Embedder, HashEmbedder

log = logging.getLogger(__name__)

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
    "bge-small-en-v1.5":                      384,
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
    kind: str = "passage",
) -> list[list[float]]:
    """Embed *texts* with caching for repeated content.

    ``kind="query"`` uses the model's query encoding when it has one (BGE models
    add a search instruction); otherwise queries and passages embed the same way.

    Cache key: (model, dimension, sha1(text)) — plus the kind for query-aware
    models. Identical texts are never re-sent to the model.
    """
    dim = dimension or _MODEL_DIMS.get(model, DEFAULT_DIMENSION)
    embedder = get_embedder(model, dim)
    as_query = kind == "query" and hasattr(embedder, "embed_queries")
    tag = f"{model}?q" if as_query else model
    results: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
    misses: list[int] = []

    for i, text in enumerate(texts):
        ckey = (tag, dim, hashlib.sha1(text.encode("utf-8")).hexdigest())
        cached = _cache.get(ckey)
        if cached is not None:
            results[i] = cached
            _cache_stats["hits"] += 1
        else:
            misses.append(i)

    if misses:
        batch = [texts[i] for i in misses]
        fresh = embedder.embed_queries(batch) if as_query else embedder.embed(batch)
        for i, vec in zip(misses, fresh):
            results[i] = vec
            ckey = (tag, dim, hashlib.sha1(texts[i].encode("utf-8")).hexdigest())
            _cache[ckey] = vec
            _cache_stats["misses"] += 1

    return results


@lru_cache(maxsize=1)
def active_embedding() -> tuple[str, int]:
    """(model, dimension) the knowledge base is indexed and searched with.

    ``settings.embedding_model``, or the hash embedder with a warning when that
    model cannot be loaded — decided once, so indexing and search always agree.
    """
    from app.config import settings

    name = settings.embedding_model or DEFAULT_MODEL
    if name == DEFAULT_MODEL:
        return DEFAULT_MODEL, DEFAULT_DIMENSION
    try:
        dim = _MODEL_DIMS[name]
        get_embedder(name, dim).embed(["warm-up"])
        return name, dim
    except Exception as exc:  # noqa: BLE001 - any load failure degrades to hash
        log.warning("embedding model %r unavailable (%s); using %s", name, exc, DEFAULT_MODEL)
        return DEFAULT_MODEL, DEFAULT_DIMENSION


def default_model(dimension: int | None = None) -> str:
    """Model for a node that names only a dimension: the active model when it matches."""
    name, dim = active_embedding()
    return name if dimension in (None, dim) else DEFAULT_MODEL


def model_for_store(dimension: int, requested: str | None = None) -> str:
    """Model to embed queries for a store whose vectors have *dimension*.

    Query and stored vectors must come from the same model: the active model
    when the dimension matches, the hash embedder for 256-d stores. A
    *requested* model that produces another dimension is a ValueError (callers
    turn it into a 422) instead of a matmul error deep in the search.
    """
    if requested:
        dim = _MODEL_DIMS.get(requested, DEFAULT_DIMENSION)
        if dim != dimension:
            raise ValueError(f"embedding model {requested!r} makes {dim}-d vectors but this store "
                             f"holds {dimension}-d vectors; omit embed_model to match the store")
        return requested
    name, dim = active_embedding()
    if dimension == dim:
        return name
    if dimension == DEFAULT_DIMENSION:
        return DEFAULT_MODEL
    raise ValueError(f"no embedding model available for a {dimension}-d store")


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


def _register_onnx() -> None:
    from app.rag.embeddings import onnx_embed

    if onnx_embed.available():
        for model_name in onnx_embed.MODELS:
            register_embedder(model_name, lambda dim, mn=model_name: onnx_embed.OnnxEmbedder(mn))


_register_sentence_tf()
_register_openai()
_register_cohere()
_register_onnx()
