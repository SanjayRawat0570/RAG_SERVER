"""Reranking API (F14).

Dedicated router for reranking operations — separate from the search router
so reranking can be applied to any hit list, not just freshly retrieved results.

Endpoints
---------
GET  /rerank/methods          List all reranking strategies with descriptions
POST /rerank                  Rerank a hit list with full explainability
POST /rerank/compare          Run multiple methods side-by-side and compare
POST /rerank/explain          Per-document scoring breakdown for one method
"""
from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import CurrentUser
from app.rag.rerank import STRATEGIES, rerank
from app.rag.rerank.cross_encoder_neural import is_model_available

router = APIRouter(prefix="/rerank", tags=["rerank"])

# ── Method catalogue ───────────────────────────────────────────────────────────

_METHOD_INFO: dict[str, dict[str, str]] = {
    "cross_encoder": {
        "name":        "Cross-Encoder (lexical)",
        "description": "Scores each document with a blend of semantic cosine similarity "
                       "and lexical keyword overlap. Fast, offline, deterministic.",
        "best_for":    "General-purpose relevance reranking without GPU/API cost.",
        "speed":       "fast",
    },
    "neural_cross_encoder": {
        "name":        "Neural Cross-Encoder",
        "description": "Uses a sentence-transformers CrossEncoder model "
                       "(cross-encoder/ms-marco-MiniLM-L-6-v2) to produce a deep "
                       "relevance score. Falls back to the lexical cross-encoder when "
                       "the model is not downloaded.",
        "best_for":    "Highest-accuracy reranking when the model weights are cached.",
        "speed":       "slow",
    },
    "mmr": {
        "name":        "Maximal Marginal Relevance (Diversity)",
        "description": "Selects results that are relevant to the query but diverse "
                       "from each other — avoiding repetitive top results.",
        "best_for":    "Exploratory queries where the user wants different perspectives.",
        "speed":       "medium",
    },
    "recency": {
        "name":        "Recency Reranking",
        "description": "Applies an exponential decay penalty based on document age. "
                       "Recent documents are boosted; old documents are pushed down.",
        "best_for":    "Time-sensitive queries ('latest', 'current', 'this year').",
        "speed":       "fast",
    },
    "authority": {
        "name":        "Authority Reranking",
        "description": "Boosts documents from trusted/authoritative sources. "
                       "Source credibility weights are fully configurable.",
        "best_for":    "Multi-source corpora where source quality differs.",
        "speed":       "fast",
    },
    "multi_factor": {
        "name":        "Multi-Factor Reranking",
        "description": "Weighted blend of relevance (cross-encoder), recency boost, "
                       "and authority weight. All weights are configurable.",
        "best_for":    "Production pipelines that need a combined signal.",
        "speed":       "medium",
    },
}


# ── Pydantic models ────────────────────────────────────────────────────────────

class HitIn(BaseModel):
    id:       str
    score:    float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


_VALID_METHODS = Literal[
    "cross_encoder", "neural_cross_encoder",
    "mmr", "recency", "authority", "multi_factor",
]


class RerankRequest(BaseModel):
    query:  str = Field(..., min_length=1)
    hits:   list[HitIn] = Field(..., min_length=1)
    method: _VALID_METHODS = "cross_encoder"
    top_n:  int | None = Field(None, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    query:   str = Field(..., min_length=1)
    hits:    list[HitIn] = Field(..., min_length=1)
    methods: list[_VALID_METHODS] = Field(
        default=["cross_encoder", "mmr", "recency"],
        min_length=1,
        max_length=6,
    )
    top_n:   int | None = Field(None, ge=1)


class ExplainRequest(BaseModel):
    query:  str = Field(..., min_length=1)
    hits:   list[HitIn] = Field(..., min_length=1)
    method: _VALID_METHODS = "cross_encoder"
    config: dict[str, Any] = Field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _hits_to_dicts(hits: list[HitIn]) -> list[dict[str, Any]]:
    return [h.model_dump() for h in hits]


def _rank_change(original_ids: list[str], reranked_ids: list[str]) -> dict[str, int]:
    orig_pos = {hid: i for i, hid in enumerate(original_ids)}
    return {
        hid: orig_pos.get(hid, len(original_ids)) - new_pos
        for new_pos, hid in enumerate(reranked_ids)
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/methods")
async def list_methods(_: CurrentUser) -> dict[str, Any]:
    """List all available reranking strategies with descriptions."""
    methods = {}
    for key, info in _METHOD_INFO.items():
        if key not in STRATEGIES:
            continue
        entry = dict(info)
        if key == "neural_cross_encoder":
            entry["model_available"] = is_model_available()
        methods[key] = entry
    return {
        "methods":       methods,
        "total":         len(methods),
        "default":       "cross_encoder",
    }


@router.post("")
async def rerank_hits(req: RerankRequest, _: CurrentUser) -> dict[str, Any]:
    """Rerank a list of hits and return explainability detail per document."""
    candidates = _hits_to_dicts(req.hits)
    original_order = [c["id"] for c in candidates]

    cfg = {**req.config}
    if req.top_n:
        cfg["top_n"] = req.top_n

    try:
        ranked = rerank(req.method, req.query, candidates, cfg)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    changes = _rank_change(original_order, [r["id"] for r in ranked])

    hits_out = []
    for new_pos, hit in enumerate(ranked):
        old_pos = original_order.index(hit["id"]) if hit["id"] in original_order else -1
        hits_out.append({
            **hit,
            "original_rank": old_pos,
            "new_rank":      new_pos,
            "rank_change":   changes.get(hit["id"], 0),
        })

    return {
        "query":  req.query,
        "method": req.method,
        "hits":   hits_out,
        "total":  len(hits_out),
        "method_info": _METHOD_INFO.get(req.method, {}),
    }


@router.post("/compare")
async def compare_methods(req: CompareRequest, _: CurrentUser) -> dict[str, Any]:
    """Run multiple reranking methods on the same hits and compare rankings."""
    candidates = _hits_to_dicts(req.hits)
    original_order = [c["id"] for c in candidates]
    cfg = {"top_n": req.top_n} if req.top_n else {}

    results: dict[str, list[str]] = {}
    for method in req.methods:
        try:
            ranked = rerank(method, req.query, candidates, cfg)
            results[method] = [r["id"] for r in ranked]
        except Exception as exc:
            results[method] = [f"error:{exc}"]

    # Agreement score: fraction of top-N positions where all methods agree.
    top_n = req.top_n or len(candidates)
    agreement_positions = [
        all(results[m][i] == results[req.methods[0]][i] for m in req.methods)
        for i in range(min(top_n, len(candidates)))
    ]
    agreement = round(sum(agreement_positions) / max(len(agreement_positions), 1), 3)

    return {
        "query":          req.query,
        "methods":        req.methods,
        "original_order": original_order,
        "rankings":       results,
        "agreement":      agreement,
        "total_hits":     len(candidates),
    }


@router.post("/explain")
async def explain_reranking(req: ExplainRequest, _: CurrentUser) -> dict[str, Any]:
    """Return a detailed per-document scoring breakdown for one method."""
    candidates = _hits_to_dicts(req.hits)
    original_order = [c["id"] for c in candidates]

    try:
        ranked = rerank(req.method, req.query, candidates, req.config)
    except ValueError as exc:
        raise HTTPException(422, str(exc))

    explanations = []
    for new_pos, hit in enumerate(ranked):
        old_pos = original_order.index(hit["id"]) if hit["id"] in original_order else -1
        explain: dict[str, Any] = {
            "id":            hit["id"],
            "original_rank": old_pos,
            "new_rank":      new_pos,
            "final_score":   round(hit.get("score", 0.0), 6),
        }
        # Method-specific explainability fields
        for key in ("rerank", "factors", "recency_boost", "authority_weight", "mmr_rank"):
            if key in hit:
                explain[key] = hit[key]

        # Human-readable verdict
        delta = old_pos - new_pos
        if delta > 0:
            explain["verdict"] = f"Moved UP {delta} positions — more relevant to query"
        elif delta < 0:
            explain["verdict"] = f"Moved DOWN {abs(delta)} positions — less relevant"
        else:
            explain["verdict"] = "No change in rank"

        explanations.append(explain)

    return {
        "query":        req.query,
        "method":       req.method,
        "method_info":  _METHOD_INFO.get(req.method, {}),
        "explanations": explanations,
        "total":        len(explanations),
    }
