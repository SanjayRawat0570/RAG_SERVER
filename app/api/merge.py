"""F4: Merge API — combine results from multiple branches/searches.

Implements the three scenarios from the spec, plus a generic merge endpoint
and Supabase audit logging.

Endpoints
---------
POST /merge/search    Scenario 1 — semantic + keyword + entity search, merged
POST /merge/answers   Scenario 2 — same question to multiple LLM providers, merged
POST /merge/chunks    Scenario 3 — order/deduplicate document chunks for narrative flow
POST /merge/run       Generic — apply any strategy to caller-supplied data
GET  /merge/strategies List all available strategies with descriptions
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.api.pipelines import STORE, DIM, build_ask_workflow, default_provider
from app.engine.executor import WorkflowExecutor
from app.engine.merging import STRATEGIES, merge
from app.models.workflow import WorkflowDef

router = APIRouter(prefix="/merge", tags=["merge"])

# ── strategy catalogue ─────────────────────────────────────────────────────────

_STRATEGY_DOCS: dict[str, str] = {
    "concat":         "Append all results into a single flat list (fastest, may duplicate).",
    "voting":         "Return the single most common value — majority rules.",
    "ranking":        "Sort by a numeric score field; keep top_n.",
    "weighted":       "Multiply each branch score by a per-branch weight, then rank.",
    "dedup":          "Remove exact duplicates while preserving first-seen order.",
    "consensus":      "Only return a value when a configured fraction of sources agree.",
    "rrf":            "Reciprocal Rank Fusion — best for combining ranked hit lists (hybrid search).",
    "answer_merge":   "Compare text answers, extract common ground, note differences, synthesize.",
    "narrative_order":"Sort document chunks by chunk_index/position for coherent reading order.",
}

# ── Supabase audit helper ──────────────────────────────────────────────────────

def _audit(user_id: str, strategy: str, input_count: int, result_count: int) -> None:
    """Write a merge audit record to Supabase; silent no-op when unconfigured."""
    try:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return
        from supabase import create_client  # type: ignore[import]
        sb = create_client(settings.supabase_url, settings.supabase_key)
        sb.table("audit_logs").insert({
            "id":           str(uuid.uuid4()),
            "user_id":      user_id,
            "operation":    "merge",
            "merge_strategy": strategy,
            "input_count":  input_count,
            "result_count": result_count,
            "created_at":   datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:  # noqa: BLE001
        pass


# ── request models ─────────────────────────────────────────────────────────────

class SearchMergeRequest(BaseModel):
    tenant: str = "default"
    query: str
    methods: list[str] = Field(
        default=["semantic", "keyword", "entity"],
        description="Search methods to run. Any subset of: semantic, keyword, entity",
    )
    strategy: str = "rrf"
    top_k: int = 10
    per_method_k: int = 5


class AnswerMergeRequest(BaseModel):
    tenant: str = "default"
    question: str
    providers: list[str] = Field(default_factory=lambda: [default_provider()])
    strategy: str = "answer_merge"


class ChunkMergeRequest(BaseModel):
    chunks: list[dict[str, Any]]
    strategy: str = "narrative_order"
    config: dict[str, Any] = Field(default_factory=dict)


class MergeRunRequest(BaseModel):
    values: list[Any]
    strategy: str
    config: dict[str, Any] = Field(default_factory=dict)


# ── helpers ────────────────────────────────────────────────────────────────────

def _search_workflow(method: str, tenant: str, query: str, top_k: int) -> WorkflowDef:
    node_type = {
        "semantic": "vector_search",
        "keyword":  "keyword_search",
        "entity":   "entity_search",
    }.get(method, "vector_search")

    return WorkflowDef(
        name=f"search_{method}",
        nodes=[
            {"id": "in",   "type": "input"},
            {"id": "search", "type": node_type,
             "config": {
                 "store":     STORE,
                 "namespace": "$.inputs.tenant",
                 "query":     "$.inputs.query",
                 "dimension": DIM,
                 "top_k":     top_k,
             }},
            {"id": "out", "type": "output", "config": {"value": "$.search"}},
        ],
        edges=[
            {"source": "in",     "target": "search"},
            {"source": "search", "target": "out"},
        ],
    )


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.get("/strategies")
async def list_strategies() -> dict:
    """List all available merge strategies with descriptions."""
    return {
        "strategies": [
            {"name": name, "description": _STRATEGY_DOCS.get(name, "")}
            for name in sorted(STRATEGIES)
        ]
    }


@router.post("/search")
async def merge_search(
    request: SearchMergeRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 1 — Multiple Search Methods merged.

    Runs up to 3 search methods (semantic, keyword, entity) in parallel,
    then merges all result lists using the requested strategy (default: rrf).
    """
    if request.strategy not in STRATEGIES:
        raise HTTPException(422, f"Unknown strategy '{request.strategy}'")

    valid_methods = {"semantic", "keyword", "entity"}
    methods = [m for m in request.methods if m in valid_methods]
    if not methods:
        raise HTTPException(422, "No valid methods specified")

    # Run each search method
    branch_results: list[list[Any]] = []
    method_counts: dict[str, int] = {}
    for method in methods:
        wf    = _search_workflow(method, request.tenant, request.query, request.per_method_k)
        run   = await WorkflowExecutor(wf).run({"tenant": request.tenant, "query": request.query})
        hits: list[Any] = run.outputs.get("out") or []
        branch_results.append(hits)
        method_counts[method] = len(hits)

    # Merge
    merged = merge(request.strategy, branch_results, {"top_n": request.top_k})
    result_count = len(merged) if isinstance(merged, list) else 1

    _audit(user["id"], request.strategy, sum(method_counts.values()), result_count)

    return {
        "query":    request.query,
        "strategy": request.strategy,
        "methods_run": methods,
        "per_method_counts": method_counts,
        "total_before_merge": sum(method_counts.values()),
        "results": merged,
        "result_count": result_count,
    }


@router.post("/answers")
async def merge_answers(
    request: AnswerMergeRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 2 — Multiple LLM Answers merged.

    Sends the same question to each requested provider, collects all answers,
    then merges them (default strategy: answer_merge) to find common ground,
    surface differences, and build a comprehensive response.
    """
    if request.strategy not in STRATEGIES:
        raise HTTPException(422, f"Unknown strategy '{request.strategy}'")

    raw_answers: list[dict[str, Any]] = []
    for provider in request.providers:
        try:
            wf  = build_ask_workflow(provider)
            run = await WorkflowExecutor(wf).run(
                {"tenant": request.tenant, "question": request.question}
            )
            out = run.outputs.get("out") or {}
            raw_answers.append({
                "provider": provider,
                "answer":   out.get("answer", ""),
                "intent":   out.get("intent"),
                "cost_usd": out.get("cost_usd"),
            })
        except Exception as exc:  # noqa: BLE001
            raw_answers.append({"provider": provider, "answer": "", "error": str(exc)})

    merged = merge(request.strategy, raw_answers, {})
    _audit(user["id"], request.strategy, len(raw_answers), 1)

    return {
        "question":       request.question,
        "strategy":       request.strategy,
        "provider_count": len(raw_answers),
        "raw_answers":    raw_answers,
        "merged":         merged,
    }


@router.post("/chunks")
async def merge_chunks(
    request: ChunkMergeRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Scenario 3 — Document Chunks combined into coherent narrative.

    Accepts a list of chunk dicts (each with ``metadata.chunk_index`` / ``position``),
    applies the requested strategy (default: narrative_order), and returns
    the ordered list plus a plain-text narrative.
    """
    if request.strategy not in STRATEGIES:
        raise HTTPException(422, f"Unknown strategy '{request.strategy}'")

    ordered = merge(request.strategy, [request.chunks], request.config)

    # Build narrative text
    narrative_parts: list[str] = []
    for chunk in (ordered if isinstance(ordered, list) else []):
        if isinstance(chunk, dict):
            text = (
                chunk.get("text")
                or chunk.get("metadata", {}).get("text")
                or ""
            )
            if text:
                narrative_parts.append(text.strip())

    _audit(user["id"], request.strategy, len(request.chunks), len(ordered) if isinstance(ordered, list) else 1)

    return {
        "strategy":       request.strategy,
        "input_count":    len(request.chunks),
        "ordered_chunks": ordered,
        "narrative":      " ".join(narrative_parts),
    }


@router.post("/run")
async def merge_run(
    request: MergeRunRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Generic merge — apply any strategy to caller-supplied data.

    Useful for custom merging outside the built-in search/answer/chunk flows.
    """
    if request.strategy not in STRATEGIES:
        raise HTTPException(422, f"Unknown strategy '{request.strategy}'. "
                                 f"Available: {sorted(STRATEGIES)}")

    result = merge(request.strategy, request.values, request.config)
    result_count = len(result) if isinstance(result, list) else 1
    _audit(user["id"], request.strategy, len(request.values), result_count)

    return {
        "strategy":     request.strategy,
        "input_count":  len(request.values),
        "result":       result,
        "result_count": result_count,
    }
