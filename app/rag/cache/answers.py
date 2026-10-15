"""Semantic answer cache for the ask endpoints (F17).

``/rag/ask`` and ``/rag/stream/ask`` (fast mode) look a question up here before
retrieval: a question whose meaning matches a recent one *in the same tenant,
for the same provider*
(cosine ≥ ``SEMANTIC_CACHE_THRESHOLD`` with the knowledge base's embedding
model) gets that complete response back — answer, citations and confidence
together, so citation markers always match their sources.

Similar meaning is not enough: "the Orders API v2 rate limit" and "how does
the v2 rate limit compare with v1" embed at 0.956, "Which team owns Project
Atlas?" and "Who leads the team that owns Project Atlas?" at 0.956. A hit also
needs the same *specifics* (:func:`specifics`): every token that carries a
digit (error codes, versions, years, SKUs), whether the question asks for a
comparison or a change, and the answer type when it is a person (who), a time
(when), a place (where) or a reason (why). The threshold (0.92) and these rules
were measured with ``eval/semantic_cache_bench.py``.

Never cached: extractive-fallback answers (a provider outage must not outlive
itself), agentic answers, and anything once the tenant's documents change
(entries carry the knowledge-base version; see ``semantic.kb_version``).
"""
from __future__ import annotations

import re
from typing import Any

from app.config import settings
from app.observability.metrics import record_cache
from app.rag.cache.semantic import SemanticCache
from app.rag.embeddings import active_embedding, embed_texts

NAME = "rag-answers"
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_./][a-z0-9]+)*")
_COMPARE = {"compare", "compared", "comparison", "vs", "versus", "differ", "difference", "differences",
            "than", "or", "between", "change", "changed", "changes"}
_ANSWER_TYPES = (("person", re.compile(r"\b(who|whom|whose)\b")),
                 ("time", re.compile(r"\b(when|what date|what time)\b")),
                 ("place", re.compile(r"\bwhere\b")),
                 ("reason", re.compile(r"\bwhy\b")))


def specifics(question: str) -> tuple[frozenset[str], bool, str]:
    """(tokens with a digit, comparison/change, answer type) — must match for a cache hit."""
    q = question.lower()
    tokens = _TOKEN.findall(q)
    kind = next((name for name, pat in _ANSWER_TYPES if pat.search(q)), "other")
    return frozenset(t for t in tokens if any(c.isdigit() for c in t)), bool(_COMPARE & set(tokens)), kind


def _cache(tenant: str, provider: str) -> SemanticCache:
    _, dim = active_embedding()
    return SemanticCache(f"{NAME}:{tenant}:{provider}", dim, settings.semantic_cache_threshold,
                         ttl=settings.semantic_cache_ttl, max_entries=settings.semantic_cache_max_entries,
                         tenant=tenant)


def _vector(question: str) -> list[float]:
    model, dim = active_embedding()
    return embed_texts([question], model, dim, kind="query")[0]


def lookup(tenant: str, question: str, provider: str) -> dict[str, Any] | None:
    """The cached response for a question meaning the same, or None."""
    if not settings.semantic_cache or not question.strip():
        return None
    payload, score = _cache(tenant, provider).lookup_scored(_vector(question))
    if payload is not None and specifics(payload.get("cached_question", "")) != specifics(question):
        payload = None  # same topic, different code/version/year or comparison
    record_cache(f"{NAME}-semantic", payload is not None)
    if payload is None:
        return None
    return {**payload, "cache_hit": True, "cache_type": "semantic", "cache_similarity": round(score, 4)}


def cacheable(out: dict[str, Any]) -> bool:
    return (out.get("provider") != "extractive-fallback"
            and not (out.get("structured") or {}).get("fallback")
            and bool(str(out.get("answer", "")).strip()))


def store(tenant: str, question: str, provider: str, out: dict[str, Any]) -> None:
    if settings.semantic_cache and question.strip() and cacheable(out):
        payload = {k: v for k, v in out.items()
                   if k not in ("event", "cache_hit", "cache_type", "cache_similarity", "run_id",
                                "duration_ms", "conversation_id", "standalone_question", "memory",
                                "cached_question")}
        payload["cached_question"] = question
        _cache(tenant, provider).put(question, _vector(question), payload)
