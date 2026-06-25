"""Hybrid search: dense + sparse fusion via Reciprocal Rank Fusion (F13).

RRF formula (Cormack et al. 2009):
    score(d) = Σ  weight_i / (k + rank_i(d))

where k = 60 (default), rank is 1-based, and weight is per-list.

RRF is ranking-only — it discards the raw scores from each retriever and
combines positions instead.  This makes it robust to very different score
scales (cosine 0–1 vs BM25 0–∞).

Usage
-----
    from app.rag.search.hybrid import hybrid_search

    hits = hybrid_search(
        query,
        store_name="docs",
        namespace="tenant-1",
        top_k=5,
    )
"""
from __future__ import annotations

from typing import Any

from app.rag.embeddings import embed_texts
from app.rag.embeddings.registry import _MODEL_DIMS, DEFAULT_DIMENSION, DEFAULT_MODEL
from app.rag.query import process_query
from app.rag.rerank.rerankers import rerank
from app.rag.search.bm25 import BM25, tokenize
from app.rag.search.semantic import extract_highlight
from app.rag.vectorstore import lookup_store


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Merge *ranked_lists* via RRF.

    Parameters
    ----------
    *ranked_lists: Each list is already sorted by descending relevance.
    k:             Constant that dampens the effect of high ranks (default 60).
    weights:       Per-list multipliers (defaults to 1.0 each).

    Returns
    -------
    Merged list sorted by descending RRF score.  Each hit carries the original
    metadata from the list where it first appeared.
    """
    n = len(ranked_lists)
    w = weights if weights and len(weights) == n else [1.0] * n

    rrf_scores: dict[str, float] = {}
    hit_meta: dict[str, dict]    = {}

    for rank_list, weight in zip(ranked_lists, w):
        for rank, hit in enumerate(rank_list, start=1):
            hid = hit["id"]
            rrf_scores[hid] = rrf_scores.get(hid, 0.0) + weight / (k + rank)
            if hid not in hit_meta:
                hit_meta[hid] = hit.get("metadata", {})

    ranked = sorted(rrf_scores.keys(), key=lambda hid: rrf_scores[hid], reverse=True)
    return [
        {"id": hid, "score": round(rrf_scores[hid], 6), "metadata": hit_meta[hid]}
        for hid in ranked
    ]


def hybrid_search(
    query: str,
    *,
    store_name: str = "default",
    namespace: str = "default",
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    embed_model: str = DEFAULT_MODEL,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
    rrf_k: int = 60,
    rerank_method: str | None = None,
    text_field: str = "text",
    expand_synonyms: dict[str, list[str]] | None = None,
    # F20 smart weighting
    weight_profile: str | None = None,   # named profile; overrides dense/sparse weights
    auto_weight: bool = False,           # auto-detect profile from query type
    explain: bool = False,               # include per-hit source attribution
) -> dict[str, Any]:
    """Hybrid dense + sparse search with RRF fusion.

    Returns the same shape as :func:`~app.rag.search.semantic.semantic_search`.
    F20 extensions: named weight profiles, auto-weighting, per-hit explain.
    """
    from app.rag.search.weights import auto_weights, get_profile, normalize_to_unit

    # Resolve weights from profile or auto-classification (F20).
    query_type = None
    if weight_profile:
        profile    = get_profile(weight_profile)
        dense_weight, sparse_weight = normalize_to_unit(
            profile.semantic_alpha, profile.keyword_alpha
        )
    elif auto_weight:
        profile, query_type = auto_weights(query)
        dense_weight, sparse_weight = normalize_to_unit(
            profile.semantic_alpha, profile.keyword_alpha
        )
        weight_profile = profile.name
    processed = process_query(query, expand_synonyms)

    store = lookup_store(store_name)
    if store is None:
        return {
            "hits": [], "query_analysis": processed,
            "total": 0, "store": store_name, "namespace": namespace,
            "error": f"Store '{store_name}' not found.",
        }

    fetch_k = top_k * 3

    # Dense retrieval
    dim = _MODEL_DIMS.get(embed_model, DEFAULT_DIMENSION)
    query_vec = embed_texts([processed["expanded_query"]], embed_model, dim)[0]
    dense_raw = store.search(query_vec, top_k=fetch_k, namespace=namespace,
                             metadata_filter=filters)
    dense_hits = [{"id": h.id, "score": h.score, "metadata": h.metadata}
                  for h in dense_raw]

    # Sparse retrieval (BM25 over in-memory records)
    sparse_hits: list[dict[str, Any]] = []
    if hasattr(store, "list_records"):
        records = store.list_records(namespace)
        if records:
            # Apply metadata pre-filter manually
            if filters:
                records = [
                    (rid, meta) for rid, meta in records
                    if all(meta.get(k) == v for k, v in filters.items())
                ]
            corpus = [tokenize(str(meta.get(text_field, ""))) for _, meta in records]
            bm25 = BM25(corpus)
            raw_scores = bm25.scores(tokenize(processed["expanded_query"]))
            ranked_idx = sorted(range(len(records)), key=lambda i: raw_scores[i], reverse=True)
            sparse_hits = [
                {"id": records[i][0], "score": float(raw_scores[i]), "metadata": records[i][1]}
                for i in ranked_idx[:fetch_k]
                if raw_scores[i] > 0
            ]

    # RRF fusion
    if sparse_hits:
        candidates = reciprocal_rank_fusion(
            dense_hits, sparse_hits,
            k=rrf_k,
            weights=[dense_weight, sparse_weight],
        )
    else:
        candidates = dense_hits  # sparse unavailable: fall back to dense-only

    # Optional reranking
    if rerank_method and candidates:
        try:
            candidates = rerank(rerank_method, query, candidates,
                                {"top_n": top_k, "text_field": text_field})
        except ValueError:
            pass

    # Build per-hit explain maps (which retriever(s) found each result).
    dense_ids  = {h["id"] for h in dense_hits}
    sparse_ids = {h["id"] for h in sparse_hits}

    def _hit_source(hid: str) -> str:
        in_dense  = hid in dense_ids
        in_sparse = hid in sparse_ids
        if in_dense and in_sparse:
            return "both"
        if in_dense:
            return "semantic"
        return "keyword"

    # Add highlights and explain.
    hits = []
    for hit in candidates[:top_k]:
        text = str(hit.get("metadata", {}).get(text_field, ""))
        entry = {**hit, "highlight": extract_highlight(text, query) if text else ""}
        if explain:
            entry["explain"] = {
                "source":         _hit_source(hit["id"]),
                "dense_weight":   round(dense_weight, 4),
                "sparse_weight":  round(sparse_weight, 4),
                "weight_profile": weight_profile or "custom",
            }
        hits.append(entry)

    result: dict[str, Any] = {
        "hits":           hits,
        "query_analysis": processed,
        "total":          len(hits),
        "store":          store_name,
        "namespace":      namespace,
        "dense_count":    len(dense_hits),
        "sparse_count":   len(sparse_hits),
        "weights": {
            "dense":   round(dense_weight, 4),
            "sparse":  round(sparse_weight, 4),
            "profile": weight_profile or "custom",
        },
    }
    if query_type:
        result["query_type"] = query_type
    return result
