"""F13: Semantic Search — HTTP API.

Endpoints
---------
POST /search/semantic       Dense vector search + optional rerank
POST /search/keyword        BM25 sparse keyword search
POST /search/hybrid         Dense + sparse RRF fusion
POST /search/rerank         Rerank an existing hit list
GET  /search/strategies     List search modes and rerank strategies
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.rag.embeddings.registry import _FACTORIES, DEFAULT_MODEL
from app.rag.query import process_query
from app.rag.rerank import STRATEGIES as RERANK_STRATEGIES, rerank
from app.rag.search import BM25, tokenize, hybrid_search, semantic_search
from app.rag.vectorstore import lookup_store

router = APIRouter(prefix="/search", tags=["search"])


# ── Request models ─────────────────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query:       str
    store:       str = "default"
    namespace:   str = "default"
    top_k:       int = Field(default=5, ge=1, le=50)
    filters:     dict[str, Any] | None = None
    embed_model: str = DEFAULT_MODEL
    rerank:      str | None = None
    rerank_top_n: int | None = None
    text_field:  str = "text"
    synonyms:    dict[str, list[str]] | None = None


class KeywordSearchRequest(BaseModel):
    query:      str
    store:      str = "default"
    namespace:  str = "default"
    top_k:      int = Field(default=5, ge=1, le=50)
    filters:    dict[str, Any] | None = None
    text_field: str = "text"
    synonyms:   dict[str, list[str]] | None = None


class HybridSearchRequest(BaseModel):
    query:          str
    store:          str = "default"
    namespace:      str = "default"
    top_k:          int = Field(default=5, ge=1, le=50)
    filters:        dict[str, Any] | None = None
    embed_model:    str = DEFAULT_MODEL
    dense_weight:   float = Field(default=0.6, ge=0.0)
    sparse_weight:  float = Field(default=0.4, ge=0.0)
    rrf_k:          int = Field(default=60, ge=1)
    rerank:         str | None = None
    text_field:     str = "text"
    synonyms:       dict[str, list[str]] | None = None
    # F20 smart weighting
    weight_profile: str | None = None   # "balanced" | "semantic" | "keyword" | "technical" | "conceptual"
    auto_weight:    bool = False         # auto-detect profile from query type
    explain:        bool = False         # include per-hit source attribution


class CompareRequest(BaseModel):
    query:     str
    store:     str = "default"
    namespace: str = "default"
    top_k:     int = Field(default=5, ge=1, le=20)
    profiles:  list[str] = Field(default=["balanced", "semantic", "keyword"],
                                  min_length=2, max_length=6)
    text_field: str = "text"


class RerankRequest(BaseModel):
    query:    str
    hits:     list[dict[str, Any]]
    method:   str = "cross_encoder"
    top_n:    int | None = None
    config:   dict[str, Any] = Field(default_factory=dict)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/strategies")
async def list_search_strategies(
    user: dict = Depends(get_current_user),
) -> dict:
    """
    List all available search modes and reranking strategies.

    Search modes
    ------------
    ``semantic`` — convert query to embedding, find nearest vectors (dense).
    ``keyword``  — BM25 term-frequency scoring (sparse).
    ``hybrid``   — combine dense + sparse via Reciprocal Rank Fusion.

    Rerank strategies
    -----------------
    ``cross_encoder`` — semantic + lexical overlap blend (quality default).
    ``mmr``           — Maximal Marginal Relevance for result diversity.
    ``recency``       — time-decay boost on a date metadata field.
    ``authority``     — per-source credibility weighting.
    ``multi_factor``  — weighted blend of relevance + recency + authority.
    """
    return {
        "search_modes": {
            "semantic": {
                "description": "Dense vector similarity search (cosine).",
                "best_for": ["natural language questions", "meaning-based queries"],
                "requires": "embedded documents in a vector store",
            },
            "keyword": {
                "description": "BM25 sparse ranking over term frequencies.",
                "best_for": ["exact terms", "product codes", "proper nouns"],
                "requires": "text stored in chunk metadata",
            },
            "hybrid": {
                "description": "Dense + sparse RRF fusion for best-of-both.",
                "best_for": ["production RAG", "mixed natural language + keywords"],
                "requires": "embedded documents + text metadata",
            },
        },
        "rerank_strategies": {
            name: {"available": True} for name in sorted(RERANK_STRATEGIES)
        },
        "embed_models": sorted(_FACTORIES.keys()),
    }


@router.post("/semantic")
async def semantic(
    request: SemanticSearchRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Dense semantic search: embed the query, find the nearest document chunks.

    The query goes through three stages:
    1. **Query processing** — intent detection, keyword extraction, synonym
       expansion (improves recall).
    2. **Embedding** — convert the expanded query to a dense vector.
    3. **Vector search** — cosine similarity against stored embeddings.
    4. **Rerank** (optional) — refine the ranked list with a more expensive
       cross-encoder or diversity (MMR) pass.

    Set ``filters`` to restrict search to a subset of documents, e.g.
    ``{"year": 2024, "format": "pdf"}``.
    """
    if request.rerank and request.rerank not in RERANK_STRATEGIES:
        raise HTTPException(422, f"Unknown rerank method '{request.rerank}'. "
                                  f"Available: {sorted(RERANK_STRATEGIES)}")
    if request.embed_model not in _FACTORIES:
        raise HTTPException(422, f"Unknown embedding model '{request.embed_model}'.")

    result = semantic_search(
        request.query,
        store_name=request.store,
        namespace=request.namespace,
        top_k=request.top_k,
        filters=request.filters,
        embed_model=request.embed_model,
        rerank_method=request.rerank,
        rerank_top_n=request.rerank_top_n,
        text_field=request.text_field,
        expand_synonyms=request.synonyms,
    )
    return result


@router.post("/keyword")
async def keyword(
    request: KeywordSearchRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Sparse BM25 keyword search over document chunk text.

    Uses Okapi BM25 (k1=1.5, b=0.75).  Only records that contain at least one
    query term are returned.  Best for exact-match queries, product codes, and
    queries where precise terminology matters.
    """
    processed = process_query(request.query, request.synonyms)

    store = lookup_store(request.store)
    if store is None:
        return {
            "hits": [], "query_analysis": processed,
            "total": 0, "store": request.store, "namespace": request.namespace,
        }

    if not hasattr(store, "list_records"):
        raise HTTPException(422, f"Store '{request.store}' does not support "
                                   "BM25 keyword search (in-memory stores only).")

    records = store.list_records(request.namespace)

    # Metadata pre-filter
    if request.filters:
        records = [
            (rid, meta) for rid, meta in records
            if all(meta.get(k) == v for k, v in request.filters.items())
        ]

    if not records:
        return {
            "hits": [], "query_analysis": processed,
            "total": 0, "store": request.store, "namespace": request.namespace,
        }

    corpus = [tokenize(str(meta.get(request.text_field, ""))) for _, meta in records]
    bm25 = BM25(corpus)
    raw_scores = bm25.scores(tokenize(processed["expanded_query"]))

    ranked_idx = sorted(range(len(records)), key=lambda i: raw_scores[i], reverse=True)
    hits = []
    for i in ranked_idx[:request.top_k]:
        if raw_scores[i] <= 0:
            break
        meta = records[i][1]
        text = str(meta.get(request.text_field, ""))
        hits.append({
            "id":        records[i][0],
            "score":     round(float(raw_scores[i]), 4),
            "metadata":  meta,
            "highlight": _keyword_highlight(text, processed["keywords"]),
        })

    return {
        "hits":           hits,
        "query_analysis": processed,
        "total":          len(hits),
        "store":          request.store,
        "namespace":      request.namespace,
    }


@router.post("/hybrid")
async def hybrid(
    request: HybridSearchRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Hybrid dense + sparse search via Reciprocal Rank Fusion (RRF).

    Combines the complementary strengths of:
    - **Dense** search (semantic meaning, handles paraphrases)
    - **Sparse** search (exact terms, product codes, rare entities)

    RRF merges the ranked lists by position rather than raw score, making it
    robust to very different score distributions between the two retrievers.

    ``dense_weight`` / ``sparse_weight`` control the relative importance.
    Higher ``rrf_k`` makes rankings more stable; lower values amplify top hits.
    """
    if request.rerank and request.rerank not in RERANK_STRATEGIES:
        raise HTTPException(422, f"Unknown rerank method '{request.rerank}'. "
                                  f"Available: {sorted(RERANK_STRATEGIES)}")
    if request.embed_model not in _FACTORIES:
        raise HTTPException(422, f"Unknown embedding model '{request.embed_model}'.")

    return hybrid_search(
        request.query,
        store_name=request.store,
        namespace=request.namespace,
        top_k=request.top_k,
        filters=request.filters,
        embed_model=request.embed_model,
        dense_weight=request.dense_weight,
        sparse_weight=request.sparse_weight,
        rrf_k=request.rrf_k,
        rerank_method=request.rerank,
        text_field=request.text_field,
        expand_synonyms=request.synonyms,
        weight_profile=request.weight_profile,
        auto_weight=request.auto_weight,
        explain=request.explain,
    )


@router.get("/profiles")
async def list_weight_profiles(_: dict = Depends(get_current_user)) -> dict:
    """List all named weight profiles for hybrid search (F20)."""
    from app.rag.search.weights import PROFILES, DEFAULT_PROFILE
    return {
        "default": DEFAULT_PROFILE,
        "profiles": {
            name: {
                "semantic_alpha": p.semantic_alpha,
                "keyword_alpha":  p.keyword_alpha,
                "description":    p.description,
            }
            for name, p in PROFILES.items()
        },
    }


@router.post("/classify")
async def classify_query_type(
    body: dict,
    _: dict = Depends(get_current_user),
) -> dict:
    """Classify a query and return the recommended weight profile (F20).

    Body: ``{"query": "..."}``
    """
    from app.rag.search.weights import PROFILES, auto_weights, classify_query
    query = body.get("query", "")
    if not query:
        from fastapi import HTTPException
        raise HTTPException(422, "query is required")
    qtype   = classify_query(query)
    profile, _ = auto_weights(query)
    return {
        "query":       query,
        "query_type":  qtype,
        "recommended_profile": profile.name,
        "weights": {
            "semantic": profile.semantic_alpha,
            "keyword":  profile.keyword_alpha,
        },
        "all_profiles": {
            name: {"semantic": p.semantic_alpha, "keyword": p.keyword_alpha}
            for name, p in PROFILES.items()
        },
    }


@router.post("/compare")
async def compare_profiles(
    request: CompareRequest,
    _: dict = Depends(get_current_user),
) -> dict:
    """Run the same query under multiple weight profiles and compare results (F20).

    Useful for tuning: see which profile surfaces the best results for a query.
    """
    from app.rag.search.weights import PROFILES

    unknown = [p for p in request.profiles if p not in PROFILES]
    if unknown:
        from fastapi import HTTPException
        raise HTTPException(422, f"Unknown profiles: {unknown}. "
                                  f"Available: {sorted(PROFILES)}")

    results = {}
    for pname in request.profiles:
        results[pname] = hybrid_search(
            request.query,
            store_name=request.store,
            namespace=request.namespace,
            top_k=request.top_k,
            text_field=request.text_field,
            weight_profile=pname,
            explain=True,
        )

    # Summarise per-profile hit IDs so the caller can see overlap.
    hit_sets = {p: {h["id"] for h in r["hits"]} for p, r in results.items()}
    all_ids  = set().union(*hit_sets.values())
    overlap  = {
        hid: [p for p, ids in hit_sets.items() if hid in ids]
        for hid in all_ids
    }

    return {
        "query":   request.query,
        "store":   request.store,
        "results": results,
        "overlap": overlap,
        "summary": {
            p: {
                "total_hits": len(r["hits"]),
                "weights":    r.get("weights", {}),
            }
            for p, r in results.items()
        },
    }


@router.post("/rerank")
async def rerank_hits(
    request: RerankRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Rerank an existing hit list with a chosen strategy.

    Use this when you already have search results (e.g. from a previous search
    or from a third-party retriever) and want to apply a reranking pass without
    re-running the full retrieval pipeline.

    Strategies
    ----------
    ``cross_encoder``  — semantic + lexical relevance blend
    ``mmr``            — promote result diversity, reduce redundancy
    ``recency``        — time-decay on a ``date`` metadata field
    ``authority``      — source credibility weights
    ``multi_factor``   — configurable blend of all three

    Pass extra config via the ``config`` field, e.g.
    ``{"lambda": 0.7}`` for MMR or ``{"half_life_days": 14}`` for recency.
    """
    if request.method not in RERANK_STRATEGIES:
        raise HTTPException(422, f"Unknown rerank method '{request.method}'. "
                                  f"Available: {sorted(RERANK_STRATEGIES)}")
    if not request.hits:
        return {"hits": [], "method": request.method, "total": 0}

    cfg = {**request.config}
    if request.top_n:
        cfg["top_n"] = request.top_n

    try:
        reranked = rerank(request.method, request.query, request.hits, cfg)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    return {
        "hits":   reranked,
        "method": request.method,
        "total":  len(reranked),
        "query":  request.query,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _keyword_highlight(text: str, keywords: list[str], max_len: int = 300) -> str:
    """Return the text snippet containing the most matching keywords."""
    if not text:
        return ""
    words = set(keywords)
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    best, best_score = text[:max_len], 0
    for sent in sentences:
        score = sum(1 for w in words if w in sent.lower())
        if score > best_score:
            best_score, best = score, sent
    return best[:max_len]
