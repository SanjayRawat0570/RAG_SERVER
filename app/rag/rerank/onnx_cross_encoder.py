"""Neural cross-encoder reranker on ONNX Runtime (F14) — no torch.

``Xenova/ms-marco-MiniLM-L-6-v2`` (~88 MB) scores each (query, passage) pair
jointly, which a bi-encoder cosine or word overlap cannot. It runs through
fastembed's ``TextCrossEncoder`` (already a dependency for the bge embedder) and
is fetched once into ``MODELS_DIR`` like the embedding model::

    python -m app.rag.rerank.onnx_cross_encoder download

Selected with ``RERANKER=onnx`` (the pipelines read :func:`pipeline_method`).
If the model cannot load, the heuristic ``cross_encoder`` is used instead and
a warning is logged once — retrieval never fails because of the reranker.
"""
from __future__ import annotations

import logging
import math
import sys
from typing import Any

from app.config import settings

log = logging.getLogger(__name__)
_model: Any = None
_failed = False


def _load() -> Any:
    global _model, _failed
    if _model is None and not _failed:
        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            try:
                _model = TextCrossEncoder(settings.reranker_model, cache_dir=settings.models_dir,
                                          local_files_only=True)
            except Exception:
                if not settings.allow_model_download:
                    raise
                from app.rag.embeddings.onnx_embed import os_trust_store

                with os_trust_store():
                    _model = TextCrossEncoder(settings.reranker_model, cache_dir=settings.models_dir)
        except Exception as exc:  # noqa: BLE001 - any load failure degrades to the heuristic
            _failed = True
            log.warning("reranker model %r unavailable (%s); using the heuristic cross_encoder",
                        settings.reranker_model, exc)
    return _model


def available() -> bool:
    return _load() is not None


def onnx_cross_encoder(query: str, candidates: list[dict], config: dict) -> list[dict]:
    model = _load()
    if model is None or not candidates:
        from app.rag.rerank.rerankers import cross_encoder

        return cross_encoder(query, candidates, config)
    field = config.get("text_field", "text")
    texts = [str(c.get("metadata", {}).get(field, c.get(field, ""))) for c in candidates]
    logits = list(model.rerank(query, texts))
    out = []
    for cand, logit in zip(candidates, logits):
        new = dict(cand)
        new["score"] = 1 / (1 + math.exp(-float(logit)))  # relevance probability
        new["rerank"] = {"model": settings.reranker_model, "logit": round(float(logit), 4),
                         "retrieval_score": cand.get("score")}
        out.append(new)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def pipeline_method() -> str:
    """Rerank strategy the ask pipelines use (``RERANKER`` setting)."""
    return "onnx_cross_encoder" if settings.reranker == "onnx" else "cross_encoder"


if __name__ == "__main__" and sys.argv[1:] == ["download"]:
    settings.allow_model_download = True
    print("ready" if available() else "failed", settings.reranker_model, settings.models_dir)
