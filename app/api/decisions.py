"""F5: Decision Trees API

Implements the four decision tree scenarios from the spec:
  1. Query type routing (exact-phrase / concept / entity / hybrid)
  2. Document type routing (PDF / DOCX / image / HTML / text)
  3. Response quality routing (confidence thresholds → action)
  4. Generic run of any pre-built decision tree
  5. Decision tree introspection (GET /decisions/tree/{name})

All decisions are logged to Supabase ``audit_logs`` with the decision path
and outcome for later analysis and improvement.

Endpoints
---------
POST /decisions/route-query    Route query to right search strategy
POST /decisions/route-document Route document to right parser
POST /decisions/route-quality  Assess confidence, choose response action
POST /decisions/run            Run a named pre-built decision workflow
GET  /decisions/tree/{name}    Return the decision tree definition as JSON
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.api.pipelines import (
    STORE, DIM,
    build_doc_type_router_workflow,
    build_search_strategy_workflow,
    build_confidence_router_workflow,
    default_provider,
)
from app.engine.executor import WorkflowExecutor
from app.models.workflow import WorkflowDef

router = APIRouter(prefix="/decisions", tags=["decisions"])

# ── pre-built decision tree registry ──────────────────────────────────────────

def _tree_registry() -> dict[str, WorkflowDef]:
    return {
        "doc_type_router":      build_doc_type_router_workflow(),
        "search_strategy_router": build_search_strategy_workflow(),
        "confidence_router":    build_confidence_router_workflow(),
    }


# ── Supabase audit helper ──────────────────────────────────────────────────────

def _audit_decision(
    user_id: str,
    tree_name: str,
    decision_path: list[str],
    outcome: str,
    confidence: float,
) -> None:
    """Log a decision record to Supabase; silent no-op when unconfigured."""
    try:
        from app.config import settings
        if not (settings.supabase_url and settings.supabase_key):
            return
        from supabase import create_client  # type: ignore[import]
        sb = create_client(settings.supabase_url, settings.supabase_key)
        sb.table("audit_logs").insert({
            "id":            str(uuid.uuid4()),
            "user_id":       user_id,
            "operation":     "decision",
            "decision_tree": tree_name,
            "decision_path": decision_path,
            "outcome":       outcome,
            "confidence":    confidence,
            "created_at":    datetime.now(timezone.utc).isoformat(),
        }).execute()
    except Exception:  # noqa: BLE001
        pass


def _extract_path(run_result: Any) -> list[str]:
    """Return the ids of nodes that ran successfully (not skipped), in order."""
    return [
        r.node_id for r in run_result.results
        if r.status not in ("skipped", "error")
    ]


# ── request / response models ──────────────────────────────────────────────────

class RouteQueryRequest(BaseModel):
    question: str
    tenant: str = "default"
    top_k: int = 5


class RouteDocumentRequest(BaseModel):
    filename: str


class RouteQualityRequest(BaseModel):
    question: str
    tenant: str = "default"


class DecisionRunRequest(BaseModel):
    tree: str = Field(..., description="One of: doc_type_router | search_strategy_router | confidence_router")
    inputs: dict[str, Any] = Field(default_factory=dict)


# ── endpoints ──────────────────────────────────────────────────────────────────

@router.post("/route-query")
async def route_query(
    request: RouteQueryRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Decision Tree: Search Strategy Selection

    Classifies the query (exact-phrase / concept / entity / hybrid) using the
    query_process intent and surface patterns, then runs the appropriate search.

    Decision path:
      input → query_process → classify → switch →
        [semantic]  use vector_search
        [keyword]   use keyword_search
        [entity]    use entity_search
        [hybrid]    use rrf_merge (all three)
    """
    from app.engine.nodes.entity_search import extract_entities

    wf  = build_search_strategy_workflow()
    run = await WorkflowExecutor(wf).run({
        "question": request.question,
        "tenant":   request.tenant,
    })

    decision     = run.outputs.get("out") or {}
    strategy     = decision.get("strategy", "hybrid")
    path         = _extract_path(run)
    classify_out = run.results
    confidence   = next(
        (r.output.get("confidence", 0.0) for r in classify_out
         if r.node_id == "classify" and isinstance(r.output, dict)),
        0.0,
    )

    # Execute the actual search chosen by the decision tree
    from app.engine.executor import WorkflowExecutor as _WE
    from app.models.workflow import WorkflowDef as _WD

    node_type_map = {
        "semantic": "vector_search",
        "keyword":  "keyword_search",
        "entity":   "entity_search",
    }

    if strategy == "hybrid":
        # Run all three and RRF-merge
        search_wf = _WD(
            name="hybrid_search",
            nodes=[
                {"id": "in",     "type": "input"},
                {"id": "dense",  "type": "vector_search",  "config": {"store": STORE, "namespace": "$.inputs.tenant", "query": "$.inputs.question", "dimension": DIM, "top_k": request.top_k}},
                {"id": "bm25",   "type": "keyword_search", "config": {"store": STORE, "namespace": "$.inputs.tenant", "query": "$.inputs.question", "dimension": DIM, "top_k": request.top_k}},
                {"id": "entity", "type": "entity_search",  "config": {"store": STORE, "namespace": "$.inputs.tenant", "query": "$.inputs.question", "dimension": DIM, "top_k": request.top_k}},
                {"id": "fuse",   "type": "merge",          "config": {"strategy": "rrf", "top_n": request.top_k}},
                {"id": "out",    "type": "output",         "config": {"value": "$.fuse"}},
            ],
            edges=[
                {"source": "in",     "target": "dense"},
                {"source": "in",     "target": "bm25"},
                {"source": "in",     "target": "entity"},
                {"source": "dense",  "target": "fuse"},
                {"source": "bm25",   "target": "fuse"},
                {"source": "entity", "target": "fuse"},
                {"source": "fuse",   "target": "out"},
            ],
        )
    else:
        node_type = node_type_map.get(strategy, "vector_search")
        search_wf = _WD(
            name=f"{strategy}_search",
            nodes=[
                {"id": "in",     "type": "input"},
                {"id": "search", "type": node_type, "config": {"store": STORE, "namespace": "$.inputs.tenant", "query": "$.inputs.question", "dimension": DIM, "top_k": request.top_k}},
                {"id": "out",    "type": "output",  "config": {"value": "$.search"}},
            ],
            edges=[
                {"source": "in",     "target": "search"},
                {"source": "search", "target": "out"},
            ],
        )

    search_run = await _WE(search_wf).run({"question": request.question, "tenant": request.tenant})
    hits = search_run.outputs.get("out") or []

    _audit_decision(user["id"], "search_strategy_router", path, strategy, confidence)

    return {
        "question":      request.question,
        "decision_tree": "search_strategy_router",
        "strategy":      strategy,
        "confidence":    confidence,
        "decision_path": path,
        "hit_count":     len(hits) if isinstance(hits, list) else 0,
        "hits":          hits,
    }


@router.post("/route-document")
async def route_document(
    request: RouteDocumentRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Decision Tree: Document Type Routing

    Inspects the file extension and routes to the correct parser.

    Decision path:
      input → classify (by extension) → switch →
        [pdf]   pypdf
        [docx]  python-docx
        [image] tesseract (OCR)
        [html]  BeautifulSoup
        [text]  generic text extraction
    """
    wf  = build_doc_type_router_workflow()
    run = await WorkflowExecutor(wf).run({"filename": request.filename.lower()})

    parser_info = run.outputs.get("out") or {}
    path        = _extract_path(run)
    classify_out = run.results
    confidence  = next(
        (r.output.get("confidence", 0.0) for r in classify_out
         if r.node_id == "classify" and isinstance(r.output, dict)),
        0.0,
    )
    category    = next(
        (r.output.get("category", "unknown") for r in classify_out
         if r.node_id == "classify" and isinstance(r.output, dict)),
        "unknown",
    )

    _audit_decision(user["id"], "doc_type_router", path, category, confidence)

    return {
        "filename":      request.filename,
        "decision_tree": "doc_type_router",
        "category":      category,
        "confidence":    confidence,
        "parser":        parser_info.get("parser"),
        "method":        parser_info.get("method"),
        "decision_path": path,
    }


@router.post("/route-quality")
async def route_quality(
    request: RouteQualityRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Decision Tree: Response Quality Routing

    Runs a vector search, assesses confidence from the top-hit score, and
    decides how the response should be handled.

    Decision path:
      input → query_process → vector_search → classify (by top score) → switch →
        [high   > 80%]  return immediately
        [medium > 60%]  return with citations
        [low    > 40%]  expand context and retry
        [none   ≤ 40%]  cannot answer confidently
    """
    wf  = build_confidence_router_workflow()
    run = await WorkflowExecutor(wf).run({
        "question": request.question,
        "tenant":   request.tenant,
    })

    action_info = run.outputs.get("out") or {}
    path        = _extract_path(run)
    classify_out = run.results
    confidence_level = next(
        (r.output.get("category", "none") for r in classify_out
         if r.node_id == "classify" and isinstance(r.output, dict)),
        "none",
    )
    confidence = next(
        (r.output.get("confidence", 0.0) for r in classify_out
         if r.node_id == "classify" and isinstance(r.output, dict)),
        0.0,
    )

    _audit_decision(user["id"], "confidence_router", path, confidence_level, confidence)

    return {
        "question":          request.question,
        "decision_tree":     "confidence_router",
        "confidence_level":  confidence_level,
        "confidence_score":  confidence,
        "action":            action_info.get("action"),
        "cite_sources":      action_info.get("cite_sources", False),
        "retry":             action_info.get("retry", False),
        "message":           action_info.get("message"),
        "decision_path":     path,
    }


@router.post("/run")
async def run_decision_tree(
    request: DecisionRunRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Run any named pre-built decision workflow with caller-supplied inputs.

    Available trees: ``doc_type_router``, ``search_strategy_router``,
    ``confidence_router``.
    """
    registry = _tree_registry()
    if request.tree not in registry:
        raise HTTPException(
            422,
            f"Unknown decision tree '{request.tree}'. "
            f"Available: {sorted(registry.keys())}",
        )

    wf  = registry[request.tree]
    run = await WorkflowExecutor(wf).run(request.inputs)

    path       = _extract_path(run)
    classify_r = next(
        (r for r in run.results if r.node_id == "classify"), None
    )
    outcome    = classify_r.output.get("category", "") if (classify_r and isinstance(classify_r.output, dict)) else ""
    confidence = classify_r.output.get("confidence", 0.0) if (classify_r and isinstance(classify_r.output, dict)) else 0.0

    _audit_decision(user["id"], request.tree, path, outcome, confidence)

    return {
        "tree":          request.tree,
        "status":        run.status,
        "outputs":       run.outputs,
        "decision_path": path,
        "outcome":       outcome,
        "confidence":    confidence,
        "duration_ms":   run.duration_ms,
    }


@router.get("/tree/{name}")
async def get_decision_tree(
    name: str,
    user: dict = Depends(get_current_user),
) -> dict:
    """
    Return the full JSON definition of a named decision tree.

    Useful for visualising the tree in the frontend or debugging routing logic.
    """
    registry = _tree_registry()
    if name not in registry:
        raise HTTPException(
            404,
            f"Decision tree '{name}' not found. "
            f"Available: {sorted(registry.keys())}",
        )

    wf = registry[name]
    return {
        "name":        wf.name,
        "description": wf.description,
        "nodes": [
            {"id": n.id, "type": n.type, "config": n.config}
            for n in wf.nodes
        ],
        "edges": [
            {"source": e.source, "target": e.target, "condition": e.condition, "label": e.label}
            for e in wf.edges
        ],
        "decision_nodes": [
            {"id": n.id, "type": n.type, "cases": n.config.get("cases", []), "default": n.config.get("default")}
            for n in wf.nodes if n.type in ("switch", "classify", "decision")
        ],
    }
