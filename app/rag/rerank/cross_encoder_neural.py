"""Neural cross-encoder reranker (F14).

Wraps sentence-transformers ``CrossEncoder`` so it slots into the same
``(query, candidates, config) -> candidates`` interface as the other strategies.

Model is loaded lazily on first use and cached in-process.  If the model
cannot be loaded (no internet, no cached weights, package error), the function
falls back silently to the cosine+lexical ``cross_encoder`` strategy already
in ``rerankers.py`` — so the rest of the pipeline always works offline.

Recommended model
-----------------
``cross-encoder/ms-marco-MiniLM-L-6-v2`` — lightweight, ~85 MB, strong
relevance signal on English retrieval tasks.

This is the production path; ``cross_encoder`` (cosine + lexical) is the dev /
offline stand-in used when no model is downloaded.
"""
from __future__ import annotations

from typing import Any

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_loaded_models: dict[str, Any] = {}


def _load_model(model_name: str):
    if model_name not in _loaded_models:
        try:
            from sentence_transformers.cross_encoder import CrossEncoder  # type: ignore[import]
            _loaded_models[model_name] = CrossEncoder(model_name)
        except Exception:
            _loaded_models[model_name] = None
    return _loaded_models[model_name]


def neural_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Rerank *candidates* using a neural cross-encoder model.

    Falls back to the lexical cross-encoder if the model is unavailable.
    """
    model_name = config.get("ce_model", _DEFAULT_MODEL)
    text_field  = config.get("text_field", "text")

    model = _load_model(model_name)

    if model is None:
        # Model not available — use offline stand-in.
        from app.rag.rerank.rerankers import cross_encoder
        return cross_encoder(query, candidates, config)

    texts = [
        str(c.get("metadata", {}).get(text_field, c.get(text_field, "")))
        for c in candidates
    ]
    pairs = [(query, doc) for doc in texts]

    try:
        raw_scores = model.predict(pairs)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()
        scores = list(raw_scores)
    except Exception:
        from app.rag.rerank.rerankers import cross_encoder
        return cross_encoder(query, candidates, config)

    # Apply sigmoid to convert logits to [0,1] probability-like scores.
    import math
    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    out = []
    for cand, raw in zip(candidates, scores):
        sig = sigmoid(float(raw))
        new = dict(cand)
        new["score"] = round(sig, 6)
        new["rerank"] = {
            "method": "neural_cross_encoder",
            "model":  model_name,
            "logit":  round(float(raw), 4),
            "score":  round(sig, 6),
        }
        out.append(new)

    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def is_model_available(model_name: str = _DEFAULT_MODEL) -> bool:
    """Return True if the neural cross-encoder model can be loaded."""
    return _load_model(model_name) is not None
