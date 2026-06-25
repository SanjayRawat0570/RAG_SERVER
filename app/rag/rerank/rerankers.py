"""Reranking strategies (F14).

All strategies are offline and deterministic so the pipeline is testable without
external services. A real cross-encoder (sentence-transformers) or an LLM
reranker implements the same ``(query, candidates, config) -> candidates``
signature and registers itself in ``STRATEGIES``.

Each candidate is a hit dict ``{"id", "score", "metadata": {"text", ...}}``.
Strategies return a re-ordered list with an updated ``score`` and a small
explainability payload.

Strategies:
* ``cross_encoder`` — relevance = semantic cosine ⊕ lexical overlap (CE stand-in)
* ``mmr``           — Maximal Marginal Relevance for diversity
* ``recency``       — exponential time decay on a date field
* ``authority``     — per-source credibility weighting
* ``multi_factor``  — weighted blend of relevance, recency, authority
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

from app.rag.embeddings import DEFAULT_DIMENSION, DEFAULT_MODEL, embed_texts


def _text(candidate: dict[str, Any], field: str = "text") -> str:
    meta = candidate.get("metadata", {})
    return str(meta.get(field, candidate.get(field, "")))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def _embeddings(query: str, candidates: list[dict], config: dict):
    model = config.get("model", DEFAULT_MODEL)
    dim = int(config.get("dimension", DEFAULT_DIMENSION))
    texts = [_text(c, config.get("text_field", "text")) for c in candidates]
    qv = np.asarray(embed_texts([query], model, dim)[0], dtype=np.float32)
    dvs = [np.asarray(v, dtype=np.float32) for v in embed_texts(texts, model, dim)]
    return qv, dvs, texts


def cross_encoder(query: str, candidates: list[dict], config: dict) -> list[dict]:
    from app.rag.search.bm25 import tokenize  # lazy — avoids circular import
    alpha = float(config.get("semantic_weight", 0.6))
    qv, dvs, texts = _embeddings(query, candidates, config)
    qtok = set(tokenize(query))
    out = []
    for cand, dv, txt in zip(candidates, dvs, texts):
        sem = _cosine(qv, dv)
        dtok = set(tokenize(txt))
        lex = (len(qtok & dtok) / len(qtok)) if qtok else 0.0
        new = dict(cand)
        new["score"] = alpha * sem + (1 - alpha) * lex
        new["rerank"] = {"semantic": round(sem, 4), "lexical": round(lex, 4)}
        out.append(new)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def mmr(query: str, candidates: list[dict], config: dict) -> list[dict]:
    lam = float(config.get("lambda", 0.5))
    qv, dvs, _ = _embeddings(query, candidates, config)
    sim_q = [_cosine(qv, dv) for dv in dvs]
    selected: list[int] = []
    remaining = list(range(len(candidates)))
    while remaining:
        best_i, best_val = remaining[0], -1e9
        for i in remaining:
            diversity = max((_cosine(dvs[i], dvs[j]) for j in selected), default=0.0)
            val = lam * sim_q[i] - (1 - lam) * diversity
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        remaining.remove(best_i)
    out = []
    for rank, i in enumerate(selected):
        new = dict(candidates[i])
        new["score"] = sim_q[i]
        new["mmr_rank"] = rank
        out.append(new)
    return out


def _age_days(value: Any, now: float) -> float | None:
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
        else:
            dt = datetime.fromisoformat(str(value))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts = dt.timestamp()
        return max(0.0, (now - ts) / 86400.0)
    except (ValueError, TypeError):
        return None


def recency(query: str, candidates: list[dict], config: dict) -> list[dict]:
    field = config.get("date_field", "date")
    half_life = float(config.get("half_life_days", 30))
    now = float(config.get("now", time.time()))
    out = []
    for cand in candidates:
        age = _age_days(cand.get("metadata", {}).get(field), now)
        boost = 0.5 ** (age / half_life) if age is not None else 1.0
        new = dict(cand)
        new["score"] = float(cand.get("score", 0.0)) * boost
        new["recency_boost"] = round(boost, 4)
        out.append(new)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def authority(query: str, candidates: list[dict], config: dict) -> list[dict]:
    field = config.get("source_field", "source")
    weights = config.get("weights", {})
    default = float(config.get("default_weight", 1.0))
    out = []
    for cand in candidates:
        src = cand.get("metadata", {}).get(field)
        weight = float(weights.get(src, default))
        new = dict(cand)
        new["score"] = float(cand.get("score", 0.0)) * weight
        new["authority_weight"] = weight
        out.append(new)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def multi_factor(query: str, candidates: list[dict], config: dict) -> list[dict]:
    """Weighted blend of relevance (cross-encoder), recency boost, authority.

    The blend weights live under ``factor_weights`` so they don't collide with
    ``weights`` (which the authority sub-strategy reads as per-source weights).
    """
    weights = config.get("factor_weights", {"relevance": 1.0, "recency": 0.0, "authority": 0.0})
    relevance = {c["id"]: c["score"] for c in cross_encoder(query, candidates, config)}
    rec = {c["id"]: c.get("recency_boost", 1.0) for c in recency(query, candidates, config)}
    auth = {c["id"]: c.get("authority_weight", 1.0) for c in authority(query, candidates, config)}
    out = []
    for cand in candidates:
        cid = cand["id"]
        score = (
            weights.get("relevance", 0.0) * relevance.get(cid, 0.0)
            + weights.get("recency", 0.0) * rec.get(cid, 1.0)
            + weights.get("authority", 0.0) * auth.get(cid, 1.0)
        )
        new = dict(cand)
        new["score"] = score
        new["factors"] = {
            "relevance": round(relevance.get(cid, 0.0), 4),
            "recency": round(rec.get(cid, 1.0), 4),
            "authority": auth.get(cid, 1.0),
        }
        out.append(new)
    out.sort(key=lambda c: c["score"], reverse=True)
    return out


def _load_neural():
    try:
        from app.rag.rerank.cross_encoder_neural import neural_cross_encoder
        return neural_cross_encoder
    except Exception:
        return None


STRATEGIES: dict[str, Any] = {
    "cross_encoder":        cross_encoder,
    "mmr":                  mmr,
    "recency":              recency,
    "authority":            authority,
    "multi_factor":         multi_factor,
}

# Register neural cross-encoder if the module loads cleanly.
_neural = _load_neural()
if _neural is not None:
    STRATEGIES["neural_cross_encoder"] = _neural


def rerank(method: str, query: str, candidates: list[dict], config: dict | None = None) -> list[dict]:
    if method not in STRATEGIES:
        raise ValueError(f"Unknown rerank method {method!r}. Available: {sorted(STRATEGIES)}")
    config = config or {}
    ranked = STRATEGIES[method](query, candidates, config)
    top_n = config.get("top_n")
    return ranked[:top_n] if top_n else ranked
