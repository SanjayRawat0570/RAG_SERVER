"""F11: Multi-Model Embedding System — HTTP API.

Endpoints
---------
GET  /embeddings/models           List all registered models with capabilities
POST /embeddings/embed            Embed one or more texts with a chosen model
POST /embeddings/similarity       Cosine similarity between two texts
POST /embeddings/select           Auto-select the best model for a document
GET  /embeddings/cache            Embedding cache performance stats
DELETE /embeddings/cache          Clear the cache
"""
from __future__ import annotations

import math
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.rag.embeddings import (
    cache_stats,
    clear_cache,
    embed_texts,
    get_embedder,
    model_info,
    select_model,
)
from app.rag.embeddings.registry import _FACTORIES, _MODEL_DIMS, DEFAULT_DIMENSION

router = APIRouter(prefix="/embeddings", tags=["embeddings"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return round(dot / (norm_a * norm_b), 6)


# ── Request models ─────────────────────────────────────────────────────────────

class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=100)
    model: str = "local-hash"
    dimension: int | None = None
    include_vectors: bool = True


class SimilarityRequest(BaseModel):
    text_a: str
    text_b: str
    model: str = "local-hash"
    dimension: int | None = None


class SelectModelRequest(BaseModel):
    content_type: str | None = None
    language: str | None = None
    quality: str = "balanced"
    sample_text: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/models")
async def list_models(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    List every embedding model the system knows about with:
    - provider, dimension, multilingual flag, best-use cases
    - ``available`` — True if the model can actually run (API key set / package installed)
    - ``registered`` — True if the factory is registered in the registry

    Use this to decide which model to pass to ``POST /embeddings/embed``.
    """
    catalogue = model_info()
    for name, info in catalogue.items():
        info["registered"] = name in _FACTORIES
    return {
        "models":    catalogue,
        "count":     len(catalogue),
        "available": [n for n, m in catalogue.items() if m.get("available")],
    }


@router.post("/embed")
async def embed(
    request: EmbedRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Embed one or more texts using the chosen model.

    Repeated texts are served from the process-wide cache — identical content
    is never re-sent to an API or re-computed locally.

    Set ``include_vectors: false`` to skip the raw numbers when you only need
    token counts or cache statistics.
    """
    if request.model not in _FACTORIES:
        raise HTTPException(
            422,
            f"Model '{request.model}' is not registered. "
            f"Available: {sorted(_FACTORIES)}"
        )
    try:
        vectors = embed_texts(request.texts, request.model, request.dimension)
    except RuntimeError as exc:
        raise HTTPException(424, str(exc)) from exc

    dim = len(vectors[0]) if vectors else (request.dimension or _MODEL_DIMS.get(request.model, DEFAULT_DIMENSION))
    result: dict[str, Any] = {
        "model":     request.model,
        "dimension": dim,
        "count":     len(vectors),
    }
    if request.include_vectors:
        result["embeddings"] = vectors
    return result


@router.post("/similarity")
async def similarity(
    request: SimilarityRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Compute cosine similarity between two texts.

    Returns a score in [-1, 1]:
    - ``1.0``   — semantically identical
    - ``0.0``   — no similarity
    - ``-1.0``  — opposite meaning

    Practical guidance: scores above 0.8 are usually strong matches;
    0.5–0.8 is moderate; below 0.5 is weak.
    """
    if request.model not in _FACTORIES:
        raise HTTPException(
            422,
            f"Model '{request.model}' is not registered. "
            f"Available: {sorted(_FACTORIES)}"
        )
    try:
        vec_a, vec_b = embed_texts(
            [request.text_a, request.text_b],
            request.model,
            request.dimension,
        )
    except RuntimeError as exc:
        raise HTTPException(424, str(exc)) from exc

    score = _cosine(vec_a, vec_b)
    interpretation = (
        "very similar" if score >= 0.85 else
        "similar"      if score >= 0.65 else
        "moderate"     if score >= 0.45 else
        "dissimilar"   if score >= 0.20 else
        "very different"
    )
    return {
        "model":          request.model,
        "score":          score,
        "interpretation": interpretation,
        "text_a_preview": request.text_a[:100],
        "text_b_preview": request.text_b[:100],
    }


@router.post("/select")
async def select_embedding_model(
    request: SelectModelRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Recommend the best embedding model for a document.

    The selector considers:
    - ``content_type``: ``"code"``, ``"text"``, ``"structured"``, ``"image"``
    - ``language``: ISO-639-1 code (e.g. ``"fr"`` for French)
    - ``quality``: ``"best"`` | ``"balanced"`` | ``"cheap"`` | ``"free"``
    - ``sample_text``: if provided, content signals are extracted automatically

    Returns the recommended model name and dimension along with the reasoning.
    """
    content_type = request.content_type
    language     = request.language

    # Auto-detect signals from sample text if provided.
    if request.sample_text and not content_type:
        txt = request.sample_text[:500]
        if any(kw in txt for kw in ("def ", "class ", "function ", "func ", "import ")):
            content_type = "code"

    model_name, dimension = select_model(
        content_type=content_type,
        language=language,
        quality=request.quality,
    )

    catalogue = model_info()
    meta = catalogue.get(model_name, {})
    return {
        "model":       model_name,
        "dimension":   dimension,
        "provider":    meta.get("provider"),
        "local":       meta.get("local", False),
        "multilingual": meta.get("multilingual", False),
        "available":   meta.get("available", False),
        "best_for":    meta.get("best_for", []),
        "signals_used": {
            "content_type": content_type,
            "language":     language,
            "quality":      request.quality,
        },
    }


@router.get("/cache")
async def get_cache_stats(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Return embedding cache performance statistics.

    ``hit_rate`` close to 1 means most embeddings are served from cache.
    High miss counts indicate lots of unique texts being embedded.
    """
    return cache_stats()


@router.delete("/cache")
async def delete_cache(
    user: dict = Depends(get_current_user),
) -> dict:
    """Clear the embedding cache and reset all hit/miss counters."""
    clear_cache()
    return {"cleared": True, "message": "Embedding cache cleared."}
