"""Semantic search pipeline (F13).

High-level function that ties query processing → embedding → vector search →
optional rerank into a single call. This is the primary entry point for
application code; the individual nodes (vector_search, keyword_search, rerank)
are the DAG-level primitives.
"""
from __future__ import annotations

import re
from typing import Any

from app.rag.embeddings import embed_texts
from app.rag.embeddings.registry import _MODEL_DIMS, DEFAULT_DIMENSION, DEFAULT_MODEL
from app.rag.query import process_query
from app.rag.rerank.rerankers import rerank
from app.rag.search.bm25 import tokenize
from app.rag.vectorstore import lookup_store


# ── Highlight extraction ───────────────────────────────────────────────────────

_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def extract_highlight(text: str, query: str, max_len: int = 300) -> str:
    """Return the sentence in *text* most lexically similar to *query*."""
    sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
    if not sentences:
        return text[:max_len]
    _STOP = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why",
             "when", "where", "who", "which", "in", "on", "of", "to", "for",
             "and", "or", "it", "this", "that", "do", "does", "did", "can"}
    qtokens = {t for t in tokenize(query) if t not in _STOP}
    if not qtokens:
        return sentences[0][:max_len] if sentences else text[:max_len]
    best, best_score = sentences[0], -1
    for sent in sentences:
        score = len(qtokens & set(tokenize(sent)))
        if score > best_score:
            best_score, best = score, sent
    return best[:max_len]


# ── Core search function ───────────────────────────────────────────────────────

def semantic_search(
    query: str,
    *,
    store_name: str = "default",
    namespace: str = "default",
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    embed_model: str = DEFAULT_MODEL,
    rerank_method: str | None = None,
    rerank_top_n: int | None = None,
    text_field: str = "text",
    expand_synonyms: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Full semantic search pipeline.

    Parameters
    ----------
    query:          Raw user question or keyword string.
    store_name:     Name of the vector store to search.
    namespace:      Partition / tenant namespace within the store.
    top_k:          Number of results to return after optional reranking.
    filters:        Exact-match metadata pre-filter (applied before scoring).
    embed_model:    Embedding model to use for the query vector.
    rerank_method:  Optional reranking strategy (see ``app.rag.rerank``).
    rerank_top_n:   Limit results after reranking (defaults to *top_k*).
    text_field:     Metadata key that holds chunk text (for highlights).
    expand_synonyms: Optional synonym map used by the query processor.

    Returns
    -------
    Dict with ``hits``, ``query_analysis``, ``total``, ``store``, ``namespace``.
    """
    # 1. Query understanding
    processed = process_query(query, expand_synonyms)

    # 2. Embed the expanded query
    dim = _MODEL_DIMS.get(embed_model, DEFAULT_DIMENSION)
    query_vec = embed_texts([processed["expanded_query"]], embed_model, dim)[0]

    # 3. Retrieve from vector store
    store = lookup_store(store_name)
    if store is None:
        return {
            "hits": [],
            "query_analysis": processed,
            "total": 0,
            "store": store_name,
            "namespace": namespace,
            "error": f"Store '{store_name}' not found.",
        }

    # Fetch more candidates when reranking to improve final quality.
    fetch_k = (top_k * 3) if rerank_method else top_k
    raw_hits = store.search(
        query_vec,
        top_k=fetch_k,
        namespace=namespace,
        metadata_filter=filters,
    )

    candidates = [
        {
            "id":    h.id,
            "score": h.score,
            "metadata": h.metadata,
        }
        for h in raw_hits
    ]

    # 4. Optional reranking
    if rerank_method and candidates:
        try:
            candidates = rerank(
                rerank_method, query, candidates,
                {"top_n": rerank_top_n or top_k, "text_field": text_field},
            )
        except ValueError:
            pass  # unknown method — skip reranking silently

    # 5. Add highlights and trim to top_k
    final_k = rerank_top_n or top_k
    hits = []
    for hit in candidates[:final_k]:
        text = str(hit.get("metadata", {}).get(text_field, ""))
        hits.append({
            **hit,
            "highlight": extract_highlight(text, query) if text else "",
        })

    return {
        "hits":           hits,
        "query_analysis": processed,
        "total":          len(hits),
        "store":          store_name,
        "namespace":      namespace,
    }
